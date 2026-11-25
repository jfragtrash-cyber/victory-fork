//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    } else if kind & MEM_COMMIT != 0 && !at.is_null() {
        // Pages of a reservation the GPU side may have found uncommitted
        // (and not looked at again for a while: `victory_gpu`'s guards).
        victory_gpu::title_memory_committed(at as usize, size);
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// A pagefile-backed section of physical pages: a whole allocation, or
/// with the pool committed ([`pool_committed`]) one [`CHUNK`] of it, made
/// the first time one of its pages is mapped.
struct Section {
    bytes: usize,
    /// Made committed if the host can.
    commit: bool,
    /// The handle (0: making it failed) and whether it is committed.
    made: std::sync::OnceLock<(usize, bool)>,
    /// A view of our own on the whole section, kept until the section goes
    /// (0: mapping it failed), and its 64 KiB pages committed through it
    /// (a bit each; a committed section needs none) ([`View::map`]).
    own: std::sync::OnceLock<usize>,
    own_committed: Mutex<Vec<u64>>,
}

impl Section {
    fn new(bytes: usize, commit: bool) -> Section {
        Section {
            bytes,
            commit,
            made: std::sync::OnceLock::new(),
            own: std::sync::OnceLock::new(),
            own_committed: Mutex::new(Vec::new()),
        }
    }

    /// Whether DirectStorage's writes into pages go through one view of our
    /// own per section, kept (`VICTORY_OWN_SECTION_VIEWS`; on by default,
    /// `0`: a view mapped and unmapped for every write, as before).
    ///
    /// Why: every request that writes into physical pages mapped a view,
    /// committed it, wrote and unmapped it again: two system calls, a commit
    /// and sixteen page faults per 64 KiB, and an unmapping that tells every
    /// processor running our threads. Run U's cutscene frames had 700-2,800
    /// such requests and 60,000 to 1.4 million page faults a frame, and the
    /// recording threads stood in the address-space lock behind them.
    fn own_views() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_OWN_SECTION_VIEWS").is_none_or(|v| v != "0"))
    }

    /// `n` pages from page `index` through the section's view of our own,
    /// committed (null if the view or a commit could not be had).
    fn own_bytes(&self, index: usize, n: usize) -> *mut u8 {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let base = *self.own.get_or_init(|| {
            let handle = self.handle();
            if handle.is_null() {
                return 0;
            }
            // SAFETY: our section; the view is kept until the section drops.
            unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, self.bytes) }.Value as usize
        });
        if base == 0 || (index + n) * PAGE > self.bytes {
            return std::ptr::null_mut();
        }
        if !self.committed() {
            let mut bits = self.own_committed.lock().unwrap_or_else(|e| e.into_inner());
            if bits.is_empty() {
                bits.resize(self.bytes.div_ceil(PAGE).div_ceil(64), 0);
            }
            for page in index..index + n {
                if bits[page / 64] & (1 << (page % 64)) == 0 {
                    if !commit(base + page * PAGE, PAGE) {
                        return std::ptr::null_mut();
                    }
                    bits[page / 64] |= 1 << (page % 64);
                }
            }
        }
        (base + index * PAGE) as *mut u8
    }

    /// The section's handle, made now if it was not yet (null if that
    /// failed).
    fn handle(&self) -> HANDLE {
        self.made.get_or_init(|| make_section(self.bytes, self.commit)).0 as HANDLE
    }

    /// Whether its pages are committed whole (no per-view commit).
    fn committed(&self) -> bool {
        self.made.get().is_some_and(|m| m.1)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        if let Some(&base) = self.own.get()
            && base != 0
        {
            use windows_sys::Win32::System::Memory::UnmapViewOfFile;
            // SAFETY: our view of the section, no longer used.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut c_void }) };
        }
        if let Some(&(handle, _)) = self.made.get()
            && handle != 0
        {
            // SAFETY: our handle; views keep the section alive on their own.
            unsafe { CloseHandle(handle as HANDLE) };
        }
    }
}

/// The pages of one committed section of the pool: 256 MiB. The title asks
/// for its 6.1 GiB pool in one call; committed whole at once a busy host
/// often refuses it (error 1450), so a committed pool is cut in chunks,
/// each committed when first mapped. A view that crosses a chunk line is
/// two views ([`XMemMapPhysicalPages`] maps runs of one section).
const CHUNK: usize = (256 << 20) / PAGE;

/// Whether the pool is made of committed chunks: only a committed section
/// can be opened as a host heap, which is what puts view memory in place
/// for the GPU side (MOONSHOT.md, phase 2). `VICTORY_POOL_COMMIT` 1 or 0
/// says; unset, whenever the GPU side puts title memory in place.
fn pool_committed() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| match std::env::var("VICTORY_POOL_COMMIT").as_deref().map(str::trim) {
        Ok("1") => true,
        Ok("0") => false,
        _ => victory_gpu::in_place_wanted(),
    });
    *ON
}

/// Makes a section of `bytes`, committed if `commit` and the host can;
/// otherwise reserved, its pages committed as views map them ([`commit`]).
/// Tells the GPU side, whose in-place heaps open only committed sections.
fn make_section(bytes: usize, commit: bool) -> (usize, bool) {
    // SAFETY: plain section creation.
    let make = |kind| unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | kind,
            (bytes as u64 >> 32) as u32,
            bytes as u32,
            std::ptr::null(),
        )
    };
    let mut handle = if commit { make(SEC_COMMIT) } else { std::ptr::null_mut() };
    let committed = !handle.is_null();
    if handle.is_null() && !on_wine() {
        if commit {
            tracing::warn!(bytes, error = %std::io::Error::last_os_error(), "a chunk of physical pages could not be committed: reserved, so its views keep host copies");
        }
        handle = make(SEC_RESERVE);
    }
    if handle.is_null() {
        tracing::warn!(bytes, error = %std::io::Error::last_os_error(), "making a section of physical pages failed");
        return (0, false);
    }
    tracing::debug!(mb = bytes >> 20, committed, "a section of physical pages made");
    victory_gpu::title_memory_section(handle as usize, bytes as u64, committed);
    (handle as usize, committed)
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Page numbers freed so far, for [`describe_pages`].
    freed: std::collections::HashSet<usize>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
    /// With [`alias_copies`]: target pages of the GPU's page-mapping copies
    /// that are memory of ours (write-watched) holding a copy of a page, by
    /// address: the page they show, or `None` once unmapped (zeros).
    copied: BTreeMap<usize, Option<Slice>>,
    /// The same by the page shown (as in `shown`): the addresses of its
    /// copies, which DirectStorage's writes into it reach ([`pages_written`]).
    copies: HashMap<(usize, usize), Vec<usize>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n.checked_mul(PAGE).is_none_or(|s| s == 0) {
        return 0;
    }
    tracing::debug!(n, flags = format_args!("{flags:#x}"), "XMemAllocatePhysicalPages");
    // Under Wine one committed section: a committed section is a sparse
    // file there, free until touched, and committing a reserved one page by
    // page costs a wineserver request each that scans every range committed
    // so far. Otherwise one reserved section whose pages are committed as
    // views map them, or committed chunks with the pool committed.
    let chunked = pool_committed() && !on_wine();
    let given = allocate(n, if chunked { CHUNK } else { n }, chunked || on_wine(), pages);
    if !given {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    given.into()
}

/// Hands out `n` pages into `pages`, from sections of `per` pages each,
/// committed if `commit`. A single section is made now, so that a failure
/// is the title's to see; chunks are made as they are first mapped.
fn allocate(n: usize, per: usize, commit: bool, pages: *mut usize) -> bool {
    let sections: Vec<Arc<Section>> = (0..n.div_ceil(per))
        .map(|k| Arc::new(Section::new(per.min(n - k * per) * PAGE, commit)))
        .collect();
    if sections.len() == 1 && sections[0].handle().is_null() {
        return false;
    }
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: sections[index / per].clone(),
            index: index % per,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    true
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        let number = unsafe { *pages.add(i) };
        physical.pages.remove(&number);
        physical.freed.insert(number);
    }
    1
}

/// What each page number of `[offset, offset + len)` of `pages` is: known,
/// freed earlier, a 4 KiB frame inside a known page, or never handed out;
/// for a destination [`write_physical`] could not write.
pub(crate) fn describe_pages(pages: &[usize], offset: usize, len: usize) -> String {
    let physical = physical();
    let end = (offset + len).div_ceil(PAGE).min(pages.len());
    let names: Vec<String> = pages[(offset / PAGE).min(end)..end]
        .iter()
        .map(|&p| {
            let what = if physical.pages.contains_key(&p) {
                "known"
            } else if physical.freed.contains(&p) {
                "freed"
            } else if physical.pages.contains_key(&(p & !(FRAMES - 1))) {
                "inside a known page"
            } else if p >= physical.next_page {
                "past every page handed out"
            } else {
                "never handed out"
            };
            format!("{p:#x} {what}")
        })
        .collect();
    format!("[{}] (next page {:#x})", names.join(", "), physical.next_page)
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// A view mapped for this alone (unmapped when dropped), not a window
    /// into the section's view of our own.
    own: bool,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        if Section::own_views() {
            let at = slice.section.own_bytes(slice.index, n);
            if !at.is_null() {
                return Some(View { at: MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() }, len: n * PAGE, own: false });
            }
        }
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle(),
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len, own: true };
        (slice.section.committed() || commit(at.Value as usize, len)).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if self.own {
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(self.at) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
    // Copies of those pages ([`alias_copies`]) take the bytes too.
    for (at, bytes) in copy_written(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Copies bytes `[offset, offset + len)` of `pages` into the copies of
/// those pages ([`alias_copies`]), from an address that shows each page
/// (or a view of our own); returns the copies' ranges written.
fn copy_written(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let mut written = Vec::new();
    let physical = physical();
    if physical.copies.is_empty() {
        return written;
    }
    let first = offset / PAGE;
    for (i, page) in pages.iter().enumerate().take(end.div_ceil(PAGE)).skip(first) {
        let Some(slice) = physical.pages.get(page) else {
            continue;
        };
        let key = slice.key(0);
        let Some(targets) = physical.copies.get(&key) else {
            continue;
        };
        let lo = if i == first { offset % PAGE } else { 0 };
        let hi = (end - i * PAGE).min(PAGE);
        let n = hi - lo;
        // The bytes: where the title shows the page, or through a view of
        // our own.
        let mut own = None;
        let source = match physical.shown.get(&key).and_then(|a| a.first()) {
            Some(&at) => at + lo,
            None => match View::map(slice, 1) {
                Some(mut view) => {
                    let at = view.bytes().as_ptr() as usize + lo;
                    own = Some(view);
                    at
                }
                None => continue,
            },
        };
        for &target in targets {
            // SAFETY: the source shows the page for as long as the table's
            // lock is held (or is our view); the target is ours.
            unsafe { std::ptr::copy_nonoverlapping(source as *const u8, (target + lo) as *mut u8, n) };
            written.push((target + lo, n));
        }
        drop(own);
    }
    written
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place there must not
    // fall into that gap (`victory_gpu::title_memory_remapping`).
    let (at, len) = touched(&physical, address as usize, n.saturating_mul(PAGE));
    let _changing = victory_gpu::title_memory_remapping(at, len);
    map_pages(&mut physical, address, n, pages)
}

/// The memory a change of `[base, base + len)` touches: that range and the
/// whole of every view it meets (`unmap` takes those away and maps what
/// stays of them again).
fn touched(physical: &Physical, base: usize, len: usize) -> (usize, usize) {
    let end = base.saturating_add(len);
    let from = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map_or(base, |(at, _)| *at);
    let to = physical
        .views
        .range(..end)
        .next_back()
        .map_or(end, |(at, (size, _))| end.max(at + size));
    (from, to - from)
}

/// [`XMemMapPhysicalPages`] with the physical pages and the remapping lock
/// held.
fn map_pages(physical: &mut Physical, address: *mut c_void, n: usize, pages: *const usize) -> *mut c_void {
    let base = address as usize;
    unmap(physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    // The physical pages' lock once for all the pages; the remapping lock
    // once a page, for the memory that page's change touches: it waits
    // only for reads in place of that memory.
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        let (at, len) = touched(&physical, address as usize, PAGE);
        let _changing = victory_gpu::title_memory_remapping(at, len);
        if map_pages(&mut physical, address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    if to.is_multiple_of(PAGE) && len.is_multiple_of(PAGE) && alias_copies() {
        return map_alias_copy(&mut physical, to, from, len);
    }
    if from != 0 && to.is_multiple_of(PAGE) && len.is_multiple_of(PAGE) && alias_runs() {
        return map_alias_runs(&mut physical, to, from, len).is_some();
    }
    if from == 0 {
        let pages = len / PAGE;
        ALIAS_COUNTS[4].fetch_add(pages as u64, std::sync::atomic::Ordering::Relaxed);
        let shown = (0..pages).filter(|k| shown_by(&physical, to + k * PAGE).is_some()).count();
        ALIAS_COUNTS[5].fetch_add(shown as u64, std::sync::atomic::Ordering::Relaxed);
    }
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let (at, span) = touched(&physical, to, len);
    let _changing = victory_gpu::title_memory_remapping(at, span);
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let Some(slice) = shown_by(&physical, from + k) else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// The physical page shown at `at`: by a view that holds it, or by a copy
/// ([`alias_copies`]).
fn shown_by(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| Slice { index: s.index + (at - base) / PAGE, section: s.section.clone() })
        .or_else(|| physical.copied.get(&(at - at % PAGE)).cloned().flatten())
}

/// Whether the GPU's page-mapping copies give their target pages a copy of
/// the page instead of a view of it (`VICTORY_ALIAS_COPY=1`; off by
/// default: views, as before).
///
/// Why: those copies are how the title streams textures, and as views they
/// cost two system calls a page (unmap, map) at 120-1,000 microseconds each
/// in the running title, and sixteen page faults when the new view is
/// first read: run R's cutscene frames of 1 to 3.5 s carried 2,000-4,800 of
/// them, 0.3-2.9 s a frame; a page done so costs 35 microseconds in a quiet
/// process (`page_mapping_cost`). On the console they change the card's
/// page tables only; the title's processor does not read the targets.
///
/// With the switch a target page becomes memory of ours, write-watched
/// (one system call the first time, `VirtualAlloc2` over its placeholder),
/// and a mapping copies the source page's 64 KiB into it; an unmapping
/// writes zeros. Everything that reads title memory reads it as before.
/// What DirectStorage writes into a page later reaches its copies
/// ([`pages_written`]). A real mapping over such a page gives it back to a
/// placeholder first ([`unmap`]).
///
/// Off by default because of what it does not carry across: a write of the
/// title's processor into the source page after the copy, and a write of
/// shaders into the copy (the two hold their own bytes until the next
/// mapping copy, where the console shows one set of bytes at both). Run U
/// with it on: the remappings cost 7-8 microseconds instead of 120-1,000,
/// and the world was drawn empty (sky and fog, no geometry): the title fills
/// pages after mapping them. Before it can be on, a copy must be taken
/// again when its source page is written (the guards see those writes).
fn alias_copies() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_COPY").is_some_and(|v| v == "1"))
}

/// Makes the placeholder page at `page` memory of ours, committed and
/// write-watched ([`alias_copies`]). False if the host refuses.
fn make_copy_page(page: usize) -> bool {
    // SAFETY: splits the page out of its placeholder (failing harmlessly
    // when it is the whole placeholder), then replaces it; both act on our
    // placeholders only.
    unsafe {
        VirtualFree(page as *mut c_void, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        !VirtualAlloc2(
            GetCurrentProcess(),
            page as *const c_void,
            PAGE,
            MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
        .is_null()
    }
}

/// [`map_alias`] with copies ([`alias_copies`]): `from` 0 unmaps. False if
/// a source page is not mapped physical memory or the host refuses.
fn map_alias_copy(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    let pages = len / PAGE;
    let mut wanted: Vec<Option<Slice>> = Vec::with_capacity(pages);
    for k in 0..pages {
        if from == 0 {
            wanted.push(None);
            continue;
        }
        let Some(slice) = shown_by(physical, from + k * PAGE) else {
            return false;
        };
        wanted.push(Some(slice));
    }
    let same = |physical: &Physical, k: usize| {
        let at = to + k * PAGE;
        match (&wanted[k], physical.copied.get(&at)) {
            (Some(w), Some(Some(now))) => Arc::ptr_eq(&now.section, &w.section) && now.index == w.index,
            (None, Some(None)) => true,
            (None, None) => shown_by(physical, at).is_none() && !physical.copied.contains_key(&at),
            _ => false,
        }
    };
    let stale: Vec<usize> = (0..pages).filter(|&k| !same(physical, k)).collect();
    let (Some(&first), Some(&last)) = (stale.first(), stale.last()) else {
        return true;
    };
    let (lo, hi) = (to + first * PAGE, to + (last + 1) * PAGE);
    let (at, span) = touched(physical, lo, hi - lo);
    let _changing = victory_gpu::title_memory_remapping(at, span);
    for k in stale {
        let page = to + k * PAGE;
        // A view there first goes (a real mapping the title made).
        if physical.views.range(..=page).next_back().is_some_and(|(b, (n, _))| page < b + n) {
            unmap(physical, page, PAGE);
        }
        // Its old page no longer reaches it.
        if let Some(Some(old)) = physical.copied.get(&page) {
            let key = old.key(0);
            if let Some(list) = physical.copies.get_mut(&key) {
                list.retain(|&a| a != page);
                if list.is_empty() {
                    physical.copies.remove(&key);
                }
            }
        }
        match &wanted[k] {
            None if !physical.copied.contains_key(&page) => continue,
            None => {
                // SAFETY: a page of ours, committed.
                unsafe { std::ptr::write_bytes(page as *mut u8, 0, PAGE) };
                physical.copied.insert(page, None);
            }
            Some(slice) => {
                if !physical.copied.contains_key(&page) && !make_copy_page(page) {
                    tracing::warn!(
                        address = format_args!("{page:#x}"),
                        error = %std::io::Error::last_os_error(),
                        "a page-mapping copy's target could not be made memory of ours"
                    );
                    return false;
                }
                // SAFETY: the source page is mapped where `from` shows it
                // (a view or a copy), the target is ours and committed.
                unsafe { std::ptr::copy_nonoverlapping((from + k * PAGE) as *const u8, page as *mut u8, PAGE) };
                physical.copies.entry(slice.key(0)).or_default().push(page);
                physical.copied.insert(page, Some(slice.clone()));
            }
        }
        victory_gpu::title_memory_remapped(page, PAGE);
    }
    true
}

/// Whether `[base, base + len)` lies in one placeholder (a reservation never
/// mapped is one already, and joining it fails).
fn one_placeholder(base: usize, len: usize) -> bool {
    let start = |at: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size; our own process.
        let n = unsafe { VirtualQueryEx(GetCurrentProcess(), at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        (n != 0 && info.State == MEM_RESERVE).then_some(info.AllocationBase as usize)
    };
    start(base).is_some_and(|first| start(base + len - 1) == Some(first))
}

/// `VirtualFree`'s flag that joins adjacent placeholders into one (not in
/// our bindings).
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// Whether [`map_alias`] leaves alone the pages that already show what they
/// are asked to, and maps runs of pages that follow one another in one
/// section as one view (`VICTORY_ALIAS_RUNS`; on by default, `0`: off).
///
/// Why: the GPU's page-mapping copies (texture streaming) unmapped and
/// mapped again every page they named, one view per page: in run N's opening
/// cutscene 12 s of `UnmapViewOfFile2` and 11 s of `MapViewOfFile3`, and in
/// run Q's frame log 3,874 remappings and 665 ms a frame there, against 4 a
/// frame in play. Each map and unmap costs some 170 microseconds (the
/// processor's other threads are told of every unmapped page).
fn alias_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_RUNS").is_none_or(|v| v != "0"))
}

/// Since the start (diagnostics): pages [`map_alias_runs`] was asked to
/// map, those that already showed their page, the views it mapped, the
/// pages asked for that showed another page; pages [`map_alias`] was asked
/// to unmap, and those of them that showed one.
static ALIAS_COUNTS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// [`map_alias`] with a source (`from` not 0), page-aligned: the source
/// pages are found first (false, with nothing changed, if one is not mapped
/// physical memory); then each run of target pages that do not already
/// show their source page is unmapped, and mapped again in runs of pages
/// consecutive in one section, one view each. The views mapped; `None` if
/// it failed.
fn map_alias_runs(physical: &mut Physical, to: usize, from: usize, len: usize) -> Option<usize> {
    use std::sync::atomic::Ordering::Relaxed;
    let pages = len / PAGE;
    let mut wanted = Vec::with_capacity(pages);
    for k in 0..pages {
        wanted.push(shown_by(physical, from + k * PAGE)?);
    }
    let same = |physical: &Physical, k: usize, w: &Slice| {
        shown_by(physical, to + k * PAGE).is_some_and(|now| Arc::ptr_eq(&now.section, &w.section) && now.index == w.index)
    };
    let stale: Vec<bool> = wanted.iter().enumerate().map(|(k, w)| !same(physical, k, w)).collect();
    let over = (0..pages).filter(|&k| stale[k] && shown_by(physical, to + k * PAGE).is_some()).count();
    ALIAS_COUNTS[3].fetch_add(over as u64, Relaxed);
    let n = ALIAS_COUNTS[0].fetch_add(pages as u64, Relaxed) + pages as u64;
    ALIAS_COUNTS[1].fetch_add(stale.iter().filter(|s| !**s).count() as u64, Relaxed);
    if n / 65536 != (n - pages as u64) / 65536 {
        tracing::info!(
            asked = n,
            already = ALIAS_COUNTS[1].load(Relaxed),
            views = ALIAS_COUNTS[2].load(Relaxed),
            over_another = ALIAS_COUNTS[3].load(Relaxed),
            unmap_asked = ALIAS_COUNTS[4].load(Relaxed),
            unmap_shown = ALIAS_COUNTS[5].load(Relaxed),
            "GPU page mappings so far: pages asked, pages that already showed their page, views mapped, pages that showed another page; pages asked unmapped, of them showing one (VICTORY_ALIAS_RUNS)"
        );
    }
    if !stale.contains(&true) {
        return Some(0);
    }
    let mut views = 0;
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let (at, span) = touched(physical, to, len);
    let _changing = victory_gpu::title_memory_remapping(at, span);
    let mut k = 0;
    while k < pages {
        if !stale[k] {
            k += 1;
            continue;
        }
        let first = k;
        while k < pages && stale[k] {
            k += 1;
        }
        let (base, run) = (to + first * PAGE, (k - first) * PAGE);
        unmap(physical, base, run);
        // The run is placeholders now, one per page or piece unmapped: made
        // one, so that a view of several pages can replace part of it.
        // SAFETY: placeholders of ours, exactly the run (unmap cut the
        // views at its ends).
        let joined = run == PAGE
            || unsafe { VirtualFree(base as *mut c_void, run, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) } != 0
            || one_placeholder(base, run);
        let mut j = first;
        while j < k {
            // Pages that follow one another in one section: one view.
            let mut end = j + 1;
            while joined
                && end < k
                && Arc::ptr_eq(&wanted[end].section, &wanted[j].section)
                && wanted[end].index == wanted[j].index + (end - j)
            {
                end += 1;
            }
            ALIAS_COUNTS[2].fetch_add(1, Relaxed);
            views += 1;
            if !map(physical, to + j * PAGE, (end - j) * PAGE, wanted[j].clone()) {
                return None;
            }
            j = end;
        }
    }
    Some(views)
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle(),
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !slice.section.committed() && !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(base, len, slice.section.handle() as usize, (slice.index * PAGE) as u64);
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Copies of pages there ([`alias_copies`]) go back to placeholders,
    // joined where several follow one another, so a view can replace them.
    let copies: Vec<usize> = physical.copied.range(base - base % PAGE..end).map(|(&a, _)| a).collect();
    if !copies.is_empty() {
        let mut run: Option<(usize, usize)> = None;
        let join = |run: Option<(usize, usize)>| {
            if let Some((from, n)) = run.filter(|&(_, n)| n > 1) {
                // SAFETY: placeholders of ours, exactly these pages.
                unsafe { VirtualFree(from as *mut c_void, n * PAGE, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) };
            }
        };
        for page in copies {
            if let Some(Some(old)) = physical.copied.remove(&page) {
                let key = old.key(0);
                if let Some(list) = physical.copies.get_mut(&key) {
                    list.retain(|&a| a != page);
                    if list.is_empty() {
                        physical.copies.remove(&key);
                    }
                }
            }
            // SAFETY: a page of ours made from a placeholder.
            unsafe { VirtualFree(page as *mut c_void, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
            victory_gpu::title_memory_remapped(page, PAGE);
            run = match run {
                Some((from, n)) if from + n * PAGE == page => Some((from, n + 1)),
                other => {
                    join(other);
                    Some((page, 1))
                }
            };
        }
        join(run);
    }
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    /// A committed pool in chunks: pages of two chunks mapped as one range
    /// are two views, each of a committed section made at the first map,
    /// and the range reads and writes as one memory.
    #[test]
    fn a_committed_pool_maps_across_chunk_lines() {
        let mut pages = [0usize; 3];
        assert!(super::allocate(3, 2, true, pages.as_mut_ptr()));
        let sections: Vec<Arc<Section>> = {
            let physical = physical();
            pages.iter().map(|p| physical.pages[p].section.clone()).collect()
        };
        assert!(Arc::ptr_eq(&sections[0], &sections[1]) && !Arc::ptr_eq(&sections[1], &sections[2]));
        assert!(sections.iter().all(|s| s.made.get().is_none()), "chunks wait for a map");
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 3 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        assert!(sections.iter().all(|s| s.committed()), "each chunk committed");
        {
            let physical = physical();
            let base = a as usize;
            assert_eq!(physical.views.get(&base).map(|v| v.0), Some(2 * PAGE));
            assert_eq!(physical.views.get(&(base + 2 * PAGE)).map(|v| v.0), Some(PAGE));
        }
        // SAFETY: all three pages are mapped read-write.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(a.cast::<u8>(), 3 * PAGE);
            bytes[2 * PAGE - 1] = 5;
            bytes[2 * PAGE] = 6;
            assert_eq!((bytes[2 * PAGE - 1], bytes[2 * PAGE]), (5, 6));
        }
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { a.cast::<u8>().add(2 * PAGE + 1).read() }, 9);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        if alias_copies() {
            // A copy unmapped is zeros (`alias_copies`).
            // SAFETY: memory of ours, committed.
            assert_eq!(unsafe { b.cast::<u8>().read() }, 0);
        } else {
            assert_eq!(query(b).host.State, MEM_RESERVE);
        }
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// The GPU's page-mapping copy of a run of pages: one view for pages
    /// consecutive in their section, nothing done for pages that already
    /// show their page, and a page changed inside such a view keeps its
    /// neighbours (`map_alias_runs`).
    #[test]
    fn page_mapping_copies_map_runs_as_one_view() {
        if alias_copies() {
            return; // views are made only without copies
        }
        let mut pages = [0usize; 4];
        let mut n = 4;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let reserve = |count: usize| XMemVirtualAlloc(std::ptr::null_mut(), count * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(4), reserve(3));
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        for k in 0..4 {
            // SAFETY: mapped read-write.
            unsafe { at(a, k).cast::<u8>().write(10 + k as u8) };
        }
        let views_at = |p: *mut c_void| physical().views.get(&(p as usize)).map(|(size, _)| *size);
        // Pages 0..3 of `a` at `b`: consecutive in their section, one view.
        assert!(map_alias(b as u64, a as u64, 3 * PAGE as u64));
        for k in 0..3 {
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { at(b, k).cast::<u8>().read() }, 10 + k as u8);
        }
        if alias_runs() {
            assert_eq!(views_at(b), Some(3 * PAGE), "one view for the run");
            // The same copy again changes nothing.
            assert_eq!(map_alias_runs(&mut physical(), b as usize, a as usize, 3 * PAGE), Some(0), "no view mapped for pages already right");
            assert_eq!(views_at(b), Some(3 * PAGE));
        }
        // The middle page now shows page 3 of `a`: its neighbours stay.
        assert!(map_alias(at(b, 1) as u64, at(a, 3) as u64, PAGE as u64));
        // SAFETY: all three mapped read-write.
        unsafe {
            assert_eq!(b.cast::<u8>().read(), 10);
            assert_eq!(at(b, 1).cast::<u8>().read(), 13);
            assert_eq!(at(b, 2).cast::<u8>().read(), 12);
            // A write through the alias shows at the source.
            at(b, 2).cast::<u8>().write(42);
            assert_eq!(at(a, 2).cast::<u8>().read(), 42);
        }
        assert!(map_alias(b as u64, 0, 3 * PAGE as u64));
        assert_eq!(query(at(b, 1)).host.State, MEM_RESERVE);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// The GPU's page-mapping copies as copies (`alias_copies`): a target
    /// page holds the source page's bytes, a new mapping copies the new
    /// page, an unmapping leaves zeros, DirectStorage's writes into the
    /// source page reach the copy, and a real mapping over a copy works.
    #[test]
    fn page_mapping_copies_copy_pages() {
        if !alias_copies() {
            return;
        }
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let reserve = |count: usize| XMemVirtualAlloc(std::ptr::null_mut(), count * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(3), reserve(2));
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: (all reads and writes below) mapped read-write or ours.
        unsafe {
            for k in 0..3 {
                at(a, k).cast::<u8>().write(20 + k as u8);
            }
            assert!(map_alias(b as u64, at(a, 1) as u64, 2 * PAGE as u64));
            assert_eq!(b.cast::<u8>().read(), 21);
            assert_eq!(at(b, 1).cast::<u8>().read(), 22);
            assert!(physical().copied.contains_key(&(b as usize)), "a copy, not a view");
            // Asked again: nothing to do.
            assert!(map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
            // Another page over the first.
            assert!(map_alias(b as u64, a as u64, PAGE as u64));
            assert_eq!(b.cast::<u8>().read(), 20);
            // DirectStorage writes into page 2 (shown at `b + PAGE`).
            assert!(write_physical(&pages[2..], 5, &[99]));
            pages_written(&pages[2..], 5, 1);
            assert_eq!(at(b, 1).cast::<u8>().add(5).read(), 99, "the write reached the copy");
            // Unmapped: zeros.
            assert!(map_alias(at(b, 1) as u64, 0, PAGE as u64));
            assert_eq!(at(b, 1).cast::<u8>().read(), 0);
            // A real mapping over both copies.
            assert_eq!(XMemMapPhysicalPages(b, 2, pages[1..].as_ptr()), b);
            assert!(physical().copied.is_empty() || !physical().copied.contains_key(&(b as usize)));
            assert_eq!(b.cast::<u8>().read(), 21);
            b.cast::<u8>().write(77);
            assert_eq!(at(a, 1).cast::<u8>().read(), 77, "a view again");
        }
        assert_eq!(XMemMapPhysicalPages(b, 2, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// What one page-mapping copy of one page costs here, unmap and map
    /// (`cargo test -p victory-hle --lib -- --ignored page_mapping_cost
    /// --nocapture`).
    #[test]
    #[ignore]
    fn page_mapping_cost() {
        let mut pages = [0usize; 64];
        let mut n = 64;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let reserve = |count: usize| XMemVirtualAlloc(std::ptr::null_mut(), count * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(64), reserve(64));
        assert_eq!(XMemMapPhysicalPages(a, 64, pages.as_ptr()), a);
        for k in 0..64 {
            // SAFETY: mapped read-write; touched so the pages are resident.
            unsafe { a.wrapping_byte_add(k * PAGE).cast::<u8>().write(k as u8) };
        }
        let rounds = 4000;
        let began = std::time::Instant::now();
        for i in 0..rounds {
            // Each round changes what `to` shows (the source moves on every cycle).
            let (to, from) = (i % 64, (i * 7 + 3 + i / 64) % 64);
            assert!(map_alias(b.wrapping_byte_add(to * PAGE) as u64, a.wrapping_byte_add(from * PAGE) as u64, PAGE as u64));
            // SAFETY: just mapped; touched as a reader would.
            assert_eq!(unsafe { b.wrapping_byte_add(to * PAGE).cast::<u8>().read() }, from as u8);
        }
        let us = began.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        println!("page mapping copy of one page: {us:.1} us (unmap, map, notifications)");
        assert!(map_alias(b as u64, 0, 64 * PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(64, pages.as_ptr()), 1);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading(first, PAGE);
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }
}

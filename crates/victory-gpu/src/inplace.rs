//! Title memory as the card's own memory: a host heap opened on memory
//! the title already has, and a buffer placed in it, so that shaders read
//! and write the title's bytes themselves and no host copy stands between
//! (the console's one memory). This file holds the two openings Direct3D
//! has, a probe that says what this PC's driver makes of them
//! (`VICTORY_INPLACE_PROBE`), and a census of which title memory could be
//! opened.
//!
//! What the Direct3D runtime and the kernel answer, measured on this PC
//! with the software adapter (the rules are theirs, not the adapter's;
//! what the card's driver adds is what the probe is for):
//!
//! - `OpenExistingHeapFromAddress` takes the base address of a whole
//!   `VirtualAlloc` allocation and nothing inside it; every page must be
//!   committed and have one protection. Opening locks every page into the
//!   PC's memory at once (256 MiB never touched: 70 ms and 256 MB of
//!   working set).
//! - `OpenExistingHeapFromAddress1` (`ID3D12Device13`) takes any committed
//!   part of an allocation, with the same rule for its pages; parts may
//!   overlap, and the whole may be opened beside them. It also takes a
//!   committed view of a section.
//! - `OpenExistingHeapFromFileMapping` takes a section made with
//!   `SEC_COMMIT` and refuses one made with `SEC_RESERVE` (how `xmem`
//!   makes the title's page pool today). The heap is the section, not a
//!   view of it: views may be unmapped, mapped elsewhere or made
//!   read-only, and the section's handle closed, with the heap alive.
//!   Opening it locks nothing (256 MiB: 0.2 ms, no working set).
//! - The heap is a custom heap in the PC's memory (write-back pages, pool
//!   L0), shared and cross-adapter shared, 64 KiB aligned. A buffer in it
//!   must have `ALLOW_CROSS_ADAPTER`; unordered access beside it is taken.
//!   A texture must be row-major with one level. Tile mappings into such
//!   a heap are refused and REMOVE THE DEVICE: never call them.
//! - While a heap opened from an address stands, the title cannot
//!   decommit, release or protect the pages in it (`VirtualFree` and
//!   `VirtualProtect` fail with error 87), nor unmap a view in it. Pages
//!   outside the opened part are free to change.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

/// The unit heaps and placed buffers are cut on, and the console's page.
const K64: usize = 64 << 10;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// `VICTORY_INPLACE_PROBE`: 0 (unset) nothing; 1 the probe on memory of
/// its own at the first sweep, and the census 30 and 150 seconds later;
/// 2 also opens one part of the title's own memory for a moment at the
/// first census.
pub(crate) fn probe_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE_PROBE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `VICTORY_INPLACE`: 0 (unset) off. 1: a window of title memory outside
/// buffers that lies in private memory committed whole is not copied: a
/// part of the title's allocation is opened as a host heap and the window
/// is a buffer placed over it ([`place_for`]). 2: title buffers too,
/// except the big ones in the card's memory.
///
/// What stage 1 covers is private memory (`XMemVirtualAlloc`'s). View
/// memory (the page pool) is turned down here: its section is made
/// reserved, which the host refuses, and what a view shows changes when
/// the title maps pages.
///
/// What it assumes, for 2: nothing of ours reads title memory that a
/// submission not yet run writes through such a buffer. With every
/// submission waited for (the default) that holds; with `VICTORY_LATE_WAIT`
/// the ranges in place are declared written like a copy's
/// (`Shadow::gpu_writes`), so the flights' rules see them, and nothing is
/// copied back for them.
///
/// shortcut: while a window is in place the title cannot decommit, free
/// or protect its pages (the call fails with error 87; the probe shows
/// it). The title's heaps are allocated once and kept, and nothing of the
/// kind was seen; [`title_memory_freeing`] is there for the runtime to
/// call before such a call once it takes them over.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| std::env::var("VICTORY_INPLACE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Whether title memory is put in place at all ([`level`] 1 or more): the
/// runtime then makes the page pool of committed sections, which alone can
/// be opened as host heaps.
pub fn in_place_wanted() -> bool {
    level() >= 1
}

/// The most title memory in place at once, in bytes
/// (`VICTORY_INPLACE_MB`, 4096): opening locks the pages into the PC's
/// memory, also those the title never touched.
fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_INPLACE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4096) << 20)
}

/// In place now: windows and their bytes, title buffers and their bytes;
/// then, since the start: asks turned down because the memory is a view
/// that cannot be opened (scattered, holed, aliased, or of a reserved
/// section) or not private memory committed whole, because the host
/// refused, because of [`most`]; and what was let go because its memory
/// changed. Then, now: bytes of private memory in place (each opening pins
/// its own pages) and bytes of the pool's sections opened (pinned whole);
/// and since the start, placements in view memory.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// Heaps opened on committed sections of the page pool, by section handle,
/// with the section's bytes: one opening per section, on first use, kept
/// while the runtime knows the section (opening pins its pages, so
/// [`most`] counts them).
///
/// shortcut: a section the title gives back stays pinned while its heap is
/// kept; the title allocates its pool once.
static SECTION_HEAPS: Mutex<Vec<(usize, ID3D12Heap, u64)>> = Mutex::new(Vec::new());

/// Of the asks turned down as memory that cannot be opened, since the
/// start: views scattered or holed, one run but shown at another address
/// too, one run of a section not committed (or over [`most`]), private
/// memory not committed whole.
static WHY_NOT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A buffer placed over title memory, and its heap.
pub(crate) struct Placed {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) heap: ID3D12Heap,
    /// The host GPU address of the title memory's first byte.
    pub(crate) gpu: u64,
    /// Whether the opening pinned pages of its own (private memory), not
    /// a part of a section heap opened once.
    pub(crate) pins: bool,
    /// Bytes from the buffer's first byte to the title memory's first.
    pub(crate) bias: u64,
}

/// Opens the 64 KiB pages that hold title memory `[start, start + len)`
/// as a heap of their own and places one buffer over all of them. `Err`
/// with nothing if the memory is not private memory committed whole
/// (nothing was asked of the host), with the host's error if it refused.
pub(crate) fn place_part(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, Option<windows::core::Error>> {
    let at = start & !(K64 as u64 - 1);
    let bytes = (start + len.max(1)).next_multiple_of(K64 as u64) - at;
    if !private_whole(at as usize, bytes as usize) {
        return Err(None);
    }
    let heap = open(device, From::Part(at as usize, bytes as usize)).map_err(Some)?;
    let buffer = place(device, &heap, 0, bytes).map_err(Some)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed { buffer, heap, gpu, pins: true, bias: start - at })
}

/// The heap of committed section `section`, opened now if it was not yet.
/// `Err(None)` if the section is not known committed, or opening it would
/// pass [`most`].
fn section_heap(device: &ID3D12Device, section: usize) -> Result<ID3D12Heap, Option<windows::core::Error>> {
    let Some(bytes) = views().sections.iter().find(|s| s.0 == section && s.2).map(|s| s.1) else {
        return Err(None);
    };
    let mut heaps = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, heap, _)) = heaps.iter().find(|h| h.0 == section) {
        return Ok(heap.clone());
    }
    if pinned() + bytes > most() {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        return Err(None);
    }
    let started = std::time::Instant::now();
    let heap = open(device, From::Section(section)).map_err(Some)?;
    STATS[9].fetch_add(bytes, Ordering::Relaxed);
    tracing::info!(
        section = format_args!("{section:#x}"),
        mb = bytes >> 20,
        ms = format_args!("{:.1}", started.elapsed().as_secs_f64() * 1e3),
        "a section of physical pages opened as a host heap: its views go in place"
    );
    heaps.push((section, heap.clone(), bytes));
    Ok(heap)
}

/// A buffer over the whole heap of committed section `section`, for the
/// raw-address directory (`crate::memory`): every view of the section is
/// the buffer at the view's offset. None unless title memory goes in place
/// and the section is known committed, no more than 2 GiB (shaders address
/// a raw view in 32 bits) and can be opened.
pub(crate) fn mirror(section: usize) -> Option<(ID3D12Resource, u64)> {
    if level() < 1 {
        return None;
    }
    let bytes = views().sections.iter().find(|s| s.0 == section && s.2)?.1;
    if bytes > 1 << 31 {
        return None;
    }
    let host = crate::host::host()?;
    let heap = section_heap(&host.device, section).ok()?;
    let buffer = place(&host.device, &heap, 0, bytes).ok()?;
    crate::host::name(&buffer, || format!("physical pages {section:#x}: the whole section in place, for shaders' raw addresses"));
    Some((buffer, bytes))
}

/// The views of `section` now: address, bytes, offset in the section.
pub(crate) fn views_of(section: usize) -> Vec<(usize, usize, u64)> {
    let v = views();
    v.shown.range((section, 0, 0)..(section.saturating_add(1), 0, 0)).map(|(&(_, offset, at), &len)| (at, len, offset)).collect()
}

/// Changes of the views since the raw-address directory last looked
/// (`crate::memory::ready`), while it is in use: a view mapped (address,
/// bytes, section, offset) or what a range showed gone (section 0).
static VIEW_CHANGES: Mutex<Vec<(usize, usize, usize, u64)>> = Mutex::new(Vec::new());
static VIEW_CHANGES_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The raw-address directory follows the views from now on.
pub(crate) fn follow_views() {
    VIEW_CHANGES_ON.store(true, Ordering::Relaxed);
}

/// The view changes since the last call, oldest first.
pub(crate) fn take_view_changes() -> Vec<(usize, usize, usize, u64)> {
    std::mem::take(&mut *VIEW_CHANGES.lock().unwrap_or_else(|e| e.into_inner()))
}

fn view_changed(at: usize, len: usize, section: usize, offset: u64) {
    if VIEW_CHANGES_ON.load(Ordering::Relaxed) {
        VIEW_CHANGES.lock().unwrap_or_else(|e| e.into_inner()).push((at, len, section, offset));
    }
}

/// Bytes of the PC's memory pinned by what is in place.
fn pinned() -> u64 {
    STATS[8].load(Ordering::Relaxed) + STATS[9].load(Ordering::Relaxed)
}

/// Title memory `[start, start + len)`, one run of committed section
/// `section` from `offset` on, as a buffer placed in the section's heap.
pub(crate) fn place_view(
    device: &ID3D12Device,
    section: usize,
    offset: u64,
    start: u64,
    len: u64,
) -> Result<Placed, Option<windows::core::Error>> {
    let heap = section_heap(device, section)?;
    // Views begin on 64 KiB lines of both the address and the section.
    let at = offset & !(K64 as u64 - 1);
    let bias = offset - at;
    debug_assert_eq!(bias, start % K64 as u64);
    let bytes = (offset + len.max(1)).next_multiple_of(K64 as u64) - at;
    let buffer = place(device, &heap, at, bytes).map_err(Some)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + bias;
    Ok(Placed { buffer, heap, gpu, pins: false, bias })
}

/// Title memory `[start, start + len)` in place, for a window outside
/// buffers or a title buffer, if [`level`] asks for it and the memory can
/// be opened; `None` leaves the caller to make a copy, as before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if level() < if window { 1 } else { 2 } {
        return None;
    }
    // View memory, known from the runtime without asking Windows: in place
    // when it is one run of one committed section that no other address
    // shows (a copy elsewhere of the same pages would not see what shaders
    // write here).
    let run = match cover(start as usize, len as usize) {
        Cover::None => None,
        Cover::Run(section, offset) if !aliased(start as usize, len as usize) => Some((section, offset)),
        cover => {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            WHY_NOT[if matches!(cover, Cover::Run(..)) { 1 } else { 0 }].fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };
    if run.is_none() && pinned() + len > most() {
        STATS[6].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let host = crate::host::host()?;
    let placed = match run {
        Some((section, offset)) => place_view(&host.device, section, offset, start, len),
        None => place_part(&host.device, start, len),
    };
    match placed {
        Ok(placed) => {
            let at = if window { 0 } else { 2 };
            STATS[at].fetch_add(1, Ordering::Relaxed);
            STATS[at + 1].fetch_add(len, Ordering::Relaxed);
            if placed.pins {
                STATS[8].fetch_add(len, Ordering::Relaxed);
            } else {
                STATS[10].fetch_add(1, Ordering::Relaxed);
            }
            Some(placed)
        }
        Err(None) => {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            WHY_NOT[if run.is_some() { 2 } else { 3 }].fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(Some(e)) => {
            if STATS[5].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = %e, "title memory not opened in place: a copy instead");
            }
            None
        }
    }
}

/// What [`place_for`] gave is gone; `pins` as it said ([`Placed::pins`]).
pub(crate) fn gone(window: bool, bytes: u64, pins: bool) {
    let at = if window { 0 } else { 2 };
    STATS[at].fetch_sub(1, Ordering::Relaxed);
    STATS[at + 1].fetch_sub(bytes, Ordering::Relaxed);
    if pins {
        STATS[8].fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// Title memory `[at, at + len)` is about to show something else, or the
/// title is about to decommit, free or protect it: what is in place there
/// is let go (the tables forget it; its heap goes when the last list and
/// slot that hold it do). Called for memory the runtime maps anew; for
/// the runtime to call before the title's own `VirtualFree` once it takes
/// that call over.
pub fn title_memory_freeing(at: usize, len: usize) {
    if STATS[0].load(Ordering::Relaxed) + STATS[2].load(Ordering::Relaxed) == 0 {
        return;
    }
    let gone = crate::shadow::in_place_let_go(at as u64, len as u64);
    STATS[7].fetch_add(gone, Ordering::Relaxed);
}

/// The line of what is in place, every ten seconds while [`level`] is on.
fn say() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    tracing::info!(
        "title memory in place: {} windows {} MB, {} title buffers {} MB (no copy, no refresh, no copy-back); turned down: {} view memory or not private memory committed whole, \
         {} refused by the host, {} over VICTORY_INPLACE_MB; {} let go because their memory changed; {} placed in view memory so far, \
         {} MB of private memory and {} MB of the pool's sections pinned; turned down as memory: {} views scattered or holed, {} shown at two addresses,          {} of a section not committed, {} private not committed whole",
        s[0],
        s[1] >> 20,
        s[2],
        s[3] >> 20,
        s[4],
        s[5],
        s[6],
        s[7],
        s[10],
        s[8] >> 20,
        s[9] >> 20,
        WHY_NOT[0].load(Ordering::Relaxed),
        WHY_NOT[1].load(Ordering::Relaxed),
        WHY_NOT[2].load(Ordering::Relaxed),
        WHY_NOT[3].load(Ordering::Relaxed)
    );
}

/// What a heap is opened on.
pub(crate) enum From {
    /// A whole `VirtualAlloc` allocation, by its base address.
    Region(usize),
    /// A committed part of one: address and bytes.
    Part(usize, usize),
    /// A section made with `SEC_COMMIT`, by its handle.
    Section(usize),
}

/// Opens a heap on memory the process has.
pub(crate) fn open(device: &ID3D12Device, from: From) -> windows::core::Result<ID3D12Heap> {
    // SAFETY: the calls take an address or a handle and check both
    // themselves; a refusal is an error code.
    unsafe {
        match from {
            From::Region(at) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromAddress(at as *const c_void),
            From::Part(at, bytes) => device.cast::<ID3D12Device13>()?.OpenExistingHeapFromAddress1(at as *const c_void, bytes),
            From::Section(handle) => device.cast::<ID3D12Device3>()?.OpenExistingHeapFromFileMapping(HANDLE(handle as *mut c_void)),
        }
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A buffer of `size` bytes at `offset` (a multiple of 64 KiB) of an
/// opened heap, which shaders may read and write, in the common state.
pub(crate) fn place(device: &ID3D12Device, heap: &ID3D12Heap, offset: u64, size: u64) -> windows::core::Result<ID3D12Resource> {
    let mut placed: Option<ID3D12Resource> = None;
    // The heap is cross-adapter shared, so the buffer must say so too.
    let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    // SAFETY: plain creation in a live heap.
    unsafe { device.CreatePlacedResource(heap, offset, &buffer_desc(size, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)? };
    placed.ok_or_else(windows::core::Error::empty)
}

// The title's views of its physical pages, as the runtime maps them: the
// census needs them to say which view memory is one run of a section (one
// placed buffer could show it) and which is scattered pages.

struct Views {
    /// By address: bytes, the section's handle, the offset in it.
    views: BTreeMap<usize, (usize, usize, u64)>,
    /// The same views by what they show: (section, offset, address) to
    /// bytes. [`aliases`] looks here, near the offsets it asks about: going
    /// through every view held the runtime's mapping lock for long while it
    /// maps pages one by one (texture refreshes waited on it for most of
    /// the slow frames), and cost the guards' every protection.
    shown: BTreeMap<(usize, u64, usize), usize>,
    /// The longest view in `shown` yet: how far before an offset a view
    /// showing it can begin.
    longest: usize,
    /// Views longer than [`BIG`], kept out of `shown` and looked at one by
    /// one (a few: the title's pool views). In `shown` the longest, a pool
    /// view of 6 GiB, made every look go through every view of its section
    /// (thousands, one per page of the GPU's page mappings): 0.5 ms of each
    /// remapping under the lock, a quarter of a second a frame (run Q).
    big: BTreeMap<(usize, u64, usize), usize>,
    /// By section and byte offset of a 64 KiB page: the address of that
    /// page in every view that shows it. A write through one view of a
    /// page is a write of the bytes every other view shows
    /// (`crate::watch` guards them all).
    pages: BTreeMap<(usize, u64), Vec<usize>>,
    /// Sections by handle: bytes, and whether made committed.
    sections: Vec<(usize, u64, bool)>,
    /// [`aliased`] by range, until the views change.
    aliased: Option<std::collections::HashMap<(usize, usize), bool>>,
}

impl Views {
    const fn new() -> Self {
        Views {
            views: BTreeMap::new(),
            shown: BTreeMap::new(),
            longest: 0,
            big: BTreeMap::new(),
            pages: BTreeMap::new(),
            sections: Vec::new(),
            aliased: None,
        }
    }

    fn insert(&mut self, at: usize, len: usize, section: usize, offset: u64) {
        self.remove(at);
        self.views.insert(at, (len, section, offset));
        if len > BIG {
            self.big.insert((section, offset, at), len);
        } else {
            self.shown.insert((section, offset, at), len);
            self.longest = self.longest.max(len);
        }
        for k in 0..len.div_ceil(K64) {
            self.pages.entry((section, offset + (k * K64) as u64)).or_default().push(at + k * K64);
        }
        self.aliased = None;
    }

    fn remove(&mut self, at: usize) {
        if let Some((len, section, offset)) = self.views.remove(&at) {
            self.shown.remove(&(section, offset, at));
            self.big.remove(&(section, offset, at));
            for k in 0..len.div_ceil(K64) {
                let key = (section, offset + (k * K64) as u64);
                if let Some(pages) = self.pages.get_mut(&key) {
                    let page = at + k * K64;
                    pages.retain(|&shown| shown != page);
                    if pages.is_empty() {
                        self.pages.remove(&key);
                    }
                }
            }
            self.aliased = None;
        }
    }
}

/// Views longer than this are [`Views::big`].
const BIG: usize = 16 << 20;

static VIEWS: Mutex<Views> = Mutex::new(Views::new());

fn views() -> std::sync::MutexGuard<'static, Views> {
    VIEWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime made a section of `bytes` for physical pages; `committed`
/// if with `SEC_COMMIT` (only such a section can be opened as a heap).
pub fn title_memory_section(section: usize, bytes: u64, committed: bool) {
    {
        let mut v = views();
        v.sections.retain(|s| s.0 != section);
        v.sections.push((section, bytes, committed));
    }
    // A handle used again: the heap opened on the section it named is not
    // this one's (what is placed in it keeps it alive).
    let mut heaps = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = heaps.iter().position(|h| h.0 == section) {
        STATS[9].fetch_sub(heaps.remove(i).2, Ordering::Relaxed);
    }
    // A view of our own made on an earlier section that had this handle's
    // number shows that section's pages, not this one's: it is forgotten
    // (left mapped: nothing reads it any more) and the next store through
    // a view of ours maps this section.
    let gone = OWN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|own| own.remove(&section)).flatten();
    if gone.is_some() {
        tracing::info!(section, "a section handle's number came round again: our own view of the earlier section is forgotten");
    }
}

/// The runtime mapped `[at, at + len)` as a view of `section` from
/// `offset` on. Call after [`crate::title_memory_remapped`] of the range.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(4, std::time::Instant::now());
    let shown = {
        let mut v = views();
        v.insert(at, len, section, offset);
        aliases_in(&v, at, len)
    };
    view_changed(at, len, section, offset);
    // What is in place at another address of the same pages is no longer
    // the only address of its pages (copies here would not see what shaders
    // write there): let go, so that the next ask makes a copy.
    for (_, alias, bytes) in shown {
        title_memory_freeing(alias, bytes);
    }
    // This view is writable from now on: a guard over the same bytes at
    // another view must look at them again, and the pages are noted written
    // there (`crate::watch`).
    crate::watch::views_mapped(section, offset, len);
}

/// What `[at, at + len)` showed is gone ([`crate::title_memory_remapped`]):
/// the views that touch it are forgotten (the runtime tells again those it
/// maps anew).
pub(crate) fn views_gone(at: usize, len: usize) {
    view_changed(at, len, 0, 0);
    let mut v = views();
    if v.views.is_empty() {
        return;
    }
    let end = at.saturating_add(len);
    let first = v.views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at).map(|(base, _)| *base);
    let gone: Vec<usize> = first.into_iter().chain(v.views.range(at.saturating_add(1)..end).map(|(base, _)| *base)).collect();
    for base in gone {
        v.remove(base);
    }
}

/// Whether physical pages that `[at, at + len)` shows are shown at another
/// address too ([`aliases`]) where the guards do not follow them: with the
/// views of a page guarded together (`crate::watch::view_guard`), a write
/// through any of them is seen, unless the page has more than
/// `crate::watch::MANY_VIEWS` views. Remembered until the views change.
///
/// Textures in such memory were hashed at every check: 124 MB a frame of
/// the 128 hashed in run N's gameplay, all of it pages shown at several
/// addresses that the guards already covered.
pub(crate) fn aliased(at: usize, len: usize) -> bool {
    let mut v = views();
    let v = &mut *v;
    if let Some(&known) = v.aliased.as_ref().and_then(|m| m.get(&(at, len))) {
        return known;
    }
    let shown = !aliases_in_parts(&v.views, &v.shown, &v.big, v.longest, at, len).is_empty();
    let unguarded = shown && (!crate::watch::view_guard() || many_views(v, at, len));
    v.aliased.get_or_insert_with(Default::default).insert((at, len), unguarded);
    unguarded
}

/// Whether a 64 KiB page that `[at, at + len)` shows is shown in more than
/// `crate::watch::MANY_VIEWS` views (whose guards are not set together).
fn many_views(v: &Views, at: usize, len: usize) -> bool {
    let end = at.saturating_add(len);
    let first = v.views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at);
    first.into_iter().chain(v.views.range(at.saturating_add(1)..end)).any(|(&base, &(bytes, section, offset))| {
        let (from, to) = (base.max(at), (base + bytes).min(end));
        let mut page = offset + (from - base) as u64;
        page -= page % K64 as u64;
        let last = offset + (to - base) as u64;
        while page < last {
            if v.pages.get(&(section, page)).is_some_and(|views| views.len() > crate::watch::MANY_VIEWS) {
                return true;
            }
            page += K64 as u64;
        }
        false
    })
}

/// The other addresses that show physical pages `[at, at + len)` shows:
/// `(address in the range, address elsewhere, bytes)`, one per overlap of
/// two views.
fn aliases_in(v: &Views, at: usize, len: usize) -> Vec<(usize, usize, usize)> {
    aliases_in_parts(&v.views, &v.shown, &v.big, v.longest, at, len)
}

fn aliases_in_parts(
    views: &BTreeMap<usize, (usize, usize, u64)>,
    shown: &BTreeMap<(usize, u64, usize), usize>,
    big: &BTreeMap<(usize, u64, usize), usize>,
    longest: usize,
    at: usize,
    len: usize,
) -> Vec<(usize, usize, usize)> {
    let end = at.saturating_add(len);
    let first = views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at);
    // The parts of sections the range shows: the view's address, the
    // section, the offsets in it, and the address of the first byte.
    let mine: Vec<(usize, usize, u64, u64, usize)> = first
        .into_iter()
        .chain(views.range(at.saturating_add(1)..end))
        .map(|(&base, &(bytes, section, offset))| {
            let (from, to) = (base.max(at), (base + bytes).min(end));
            (base, section, offset + (from - base) as u64, offset + (to - base) as u64, from)
        })
        .collect();
    let mut found = Vec::new();
    if mine.is_empty() {
        return found;
    }
    for &(own, section, from, to, address) in &mine {
        // The views of the section that begin less than the longest view
        // before `from` and before `to`: those that may show part of it.
        let low = (section, from.saturating_sub(longest as u64), 0);
        let near = shown.range(low..(section, to, 0));
        let long = big.range((section, 0, 0)..(section, to, 0));
        for (&(_, offset, base), &bytes) in near.chain(long) {
            let (lo, hi) = (from.max(offset), to.min(offset + bytes as u64));
            if base != own && lo < hi {
                found.push((address + (lo - from) as usize, base + (lo - offset) as usize, (hi - lo) as usize));
            }
        }
    }
    found
}

/// The title views that overlap `[start, end)`, as `(address, bytes)`.
pub(crate) fn views_between(start: usize, end: usize) -> Vec<(usize, usize)> {
    if start >= end {
        return Vec::new();
    }
    let v = views();
    let mut out = Vec::new();
    if let Some((&base, &(bytes, _, _))) = v.views.range(..=start).next_back()
        && base + bytes > start
    {
        out.push((base, bytes));
    }
    for (&base, &(bytes, _, _)) in v.views.range(start.saturating_add(1)..end) {
        out.push((base, bytes));
    }
    out
}

/// The address of the 64 KiB page shown at `page` in every title view that
/// shows it, `page`'s own address included; empty if `page` is not a view.
pub(crate) fn pages_of(page: usize) -> Vec<usize> {
    let v = views();
    let Some((&base, &(bytes, section, offset))) = v.views.range(..=page).next_back() else {
        return Vec::new();
    };
    let within = page.wrapping_sub(base);
    if within >= bytes || !within.is_multiple_of(K64) {
        return Vec::new();
    }
    v.pages.get(&(section, offset + within as u64)).cloned().unwrap_or_default()
}

/// The title views that show any of the physical bytes `[offset, offset +
/// len)` of `section`, as `(address, bytes)` ranges in address order; not
/// those of a page shown in very many views (`MANY_VIEWS`).
pub(crate) fn views_of_section(section: usize, offset: u64, len: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let v = views();
    let end = offset.saturating_add(len as u64);
    let mut pages: Vec<usize> = Vec::new();
    let mut at = offset - offset % K64 as u64;
    while at < end {
        if let Some(shown) = v.pages.get(&(section, at))
            && shown.len() <= crate::watch::MANY_VIEWS
        {
            pages.extend(shown.iter().copied());
        }
        match at.checked_add(K64 as u64) {
            Some(next) => at = next,
            None => break,
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(pages.len());
    for page in pages {
        match runs.last_mut() {
            Some((first, bytes)) if *first + *bytes == page => *bytes += K64,
            _ => runs.push((page, K64)),
        }
    }
    runs
}

/// Whether what shaders wrote goes back into title view memory through a
/// view of our own (`VICTORY_BACK_OWN_VIEW=1`; off by default).
///
/// The copy-back writes title memory at the title's address, so it opens
/// the guard on the 64 KiB block first (and on every other view of the
/// page), and the block then counts as written by the title: the host
/// copy that just put the bytes there copies all 64 KiB of it back in at
/// its next refresh, and sends them up to the card again if it lives
/// there. In runs 218-222 the bytes copied in went with the bytes copied
/// back, 0.83 MB for each MB (78 readings), some 50 MB of 142 MB a frame.
///
/// With the switch the bytes are stored through a view of our own on the
/// same section, which no guard covers: the title's views stay read-only
/// (a title write still faults and is seen), the write-watch journal is
/// told the page was written (`crate::watch::wrote_own`) so that every
/// other copy of it takes it again, and the copy that wrote it notes the
/// page current (`put_back_own` in `crate::shadow`).
///
/// `VICTORY_BACK_OWN_VIEW=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log, for measuring it in one scene.
pub(crate) fn back_own() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_BACK_OWN_VIEW").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "copy-back through views of our own from now (VICTORY_BACK_OWN_VIEW=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// A view of our own on a whole section of the title's physical pages:
/// where it is, its bytes, and which of its 64 KiB blocks we have asked
/// to be committed through it (a bit each).
struct Own {
    base: usize,
    bytes: u64,
    committed: Vec<u64>,
}

/// Our own views by section handle; `None` in a section's place if the
/// host refused the view (not asked again).
static OWN: Mutex<Option<std::collections::HashMap<usize, Option<Own>>>> = Mutex::new(None);

/// Stores the parts `[from, to)` of `bytes` (offsets in it, in order) at
/// title address `at` on, through a view of our own ([`back_own`]), if
/// `[at, at + bytes.len())` lies in one 64 KiB page of a title view of a
/// section. Gives the addresses of that page in every title view that shows
/// it (the page of `at` among them), or `None` with nothing written: not
/// view memory, a page of very many views, the runtime changing mappings at
/// this moment, or the host refusing.
pub(crate) fn write_own(at: usize, bytes: &[u8], parts: &[(usize, usize)]) -> Option<Vec<usize>> {
    if bytes.is_empty() || at / K64 != (at + bytes.len() - 1) / K64 {
        return None;
    }
    // No mapping changes until the bytes are in: the view found below
    // shows these pages for as long as this is held.
    let _still = crate::watch::title_memory_reading_now(at, bytes.len())?;
    let (section, offset, shown, section_bytes) = {
        let v = views();
        let (&base, &(len, section, offset)) = v.views.range(..=at).next_back()?;
        if at + bytes.len() > base + len {
            return None;
        }
        let offset = offset + (at - base) as u64;
        let shown = v.pages.get(&(section, offset - offset % K64 as u64))?.clone();
        if shown.len() > crate::watch::MANY_VIEWS {
            return None;
        }
        let section_bytes = v.sections.iter().find(|s| s.0 == section)?.1;
        (section, offset, shown, section_bytes)
    };
    if offset + bytes.len() as u64 > section_bytes {
        return None;
    }
    let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
    let view = own
        .get_or_insert_with(Default::default)
        .entry(section)
        .or_insert_with(|| {
            // SAFETY: a section handle the runtime gave us, alive as long
            // as the title's views of it are; the view is ours and stays.
            let at = unsafe { mem::MapViewOfFile(section as *mut c_void, mem::FILE_MAP_WRITE, 0, 0, 0) };
            if at.Value.is_null() {
                tracing::warn!(
                    section,
                    error = %std::io::Error::last_os_error(),
                    "no view of our own on a section of title pages: what shaders wrote goes back the old way there"
                );
                return None;
            }
            tracing::info!(
                section,
                mb = section_bytes >> 20,
                "a view of our own on a section of title pages, for what shaders wrote (VICTORY_BACK_OWN_VIEW=1)"
            );
            Some(Own {
                base: at.Value as usize,
                bytes: section_bytes,
                committed: vec![0; (section_bytes as usize).div_ceil(K64).div_ceil(64)],
            })
        })
        .as_mut()?;
    if offset + bytes.len() as u64 > view.bytes {
        return None;
    }
    let block = offset as usize / K64;
    if (view.committed[block / 64] & (1u64 << (block % 64))) == 0 {
        // The title's view committed these pages of the section; asked
        // once more through ours, which changes nothing if they are.
        // SAFETY: a block inside our own view.
        let there = unsafe {
            mem::VirtualAlloc((view.base + block * K64) as *const c_void, K64, mem::MEM_COMMIT, mem::PAGE_READWRITE)
        };
        if there.is_null() {
            return None;
        }
        view.committed[block / 64] |= 1u64 << (block % 64);
    }
    // The bytes go in outside the lock: the merges of several threads
    // (`crate::shadow::relay_wide`) each store a page of their own, and
    // under it they stored one after the other.
    let into = view.base + offset as usize;
    drop(own);
    // SAFETY: committed bytes inside our own view, which is never unmapped
    // (one forgotten stays mapped, `title_memory_section`); the source is
    // the caller's slice, the parts inside it.
    for &(from, to) in parts {
        let to = to.min(bytes.len());
        if from < to {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr().add(from), (into + from) as *mut u8, to - from) };
        }
    }
    Some(shown)
}

/// How the views cover `[at, at + len)`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Cover {
    /// No view holds its first byte.
    None,
    /// Views hold a part of it only.
    Holes,
    /// Views hold all of it, from more than one place.
    Scattered,
    /// All of it is one run of one section: the section and the offset of
    /// `at` in it.
    Run(usize, u64),
}

fn cover_in(views: &BTreeMap<usize, (usize, usize, u64)>, at: usize, len: usize) -> Cover {
    let end = at + len.max(1);
    let Some((&base, &(bytes, section, offset))) = views.range(..=at).next_back().filter(|(base, view)| *base + view.0 > at) else {
        return Cover::None;
    };
    let first = (section, offset + (at - base) as u64);
    let (mut reached, mut one) = (base + bytes, true);
    while reached < end {
        let Some(&(bytes, s, o)) = views.get(&reached) else {
            return Cover::Holes;
        };
        one &= s == first.0 && o == first.1 + (reached - at) as u64;
        reached += bytes;
    }
    if one { Cover::Run(first.0, first.1) } else { Cover::Scattered }
}

/// [`Cover`] of `[at, at + len)` by the title's views as they are now.
pub(crate) fn cover(at: usize, len: usize) -> Cover {
    cover_in(&views().views, at, len)
}

/// Whether `[at, at + len)` is private memory, all committed and
/// read-write: what [`From::Part`] takes. One question to Windows (it
/// walks the page tables to the end of the pages that share the first
/// one's state: a millisecond in the title's big blocks, so ask when a
/// copy is made, not when it is used).
pub(crate) fn private_whole(at: usize, len: usize) -> bool {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == mem::MEM_COMMIT
        && info.Type == mem::MEM_PRIVATE
        && info.Protect == mem::PAGE_READWRITE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at.saturating_add(len)
}

fn hr<T>(result: &windows::core::Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_owned(),
        Err(e) => format!("REFUSED {:#010x} ({})", e.code().0, e.message().trim()),
    }
}

/// A queue, a list and a fence of the probe's own on `device`: nothing of
/// the host's submissions is touched.
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig { device: device.clone(), queue, allocator, list, fence, done: 0, event })
        }
    }

    /// Records with `record`, runs it and waits (ten seconds at most).
    fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 10_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }

    fn committed(&self, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE), state, None, &mut buffer)?
        };
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// What the GPU reads of `size` bytes of `resource` from `offset`.
    fn fetch(&mut self, resource: &ID3D12Resource, offset: u64, size: usize) -> windows::core::Result<Vec<u8>> {
        let back = self.committed(D3D12_HEAP_TYPE_READBACK, size as u64, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(&back, 0, resource, offset, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size).to_vec();
            back.Unmap(0, None);
            Ok(bytes)
        }
    }

    /// The GPU writes `bytes` at `offset` of `resource` (a copy).
    fn store(&mut self, resource: &ID3D12Resource, offset: u64, bytes: &[u8]) -> windows::core::Result<()> {
        let up = self.committed(D3D12_HEAP_TYPE_UPLOAD, bytes.len() as u64, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's bytes until unmapped.
        unsafe {
            up.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: two live buffers, the range inside both.
        self.run(|list| unsafe { list.CopyBufferRegion(resource, offset, &up, 0, bytes.len() as u64) })
    }

    /// A compute shader stores, through a root unordered-access view at
    /// `gpu`, the sum of the buffer's first two words in its third.
    fn shade(&mut self, gpu: u64) -> windows::core::Result<()> {
        const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(1, 1, 1)]\nvoid main() { b.Store(8, b.Load(0) + b.Load(4)); }\n";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("inplace"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        // SAFETY: an open list of ours; the view names a live buffer.
        self.run(|list| unsafe {
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootUnorderedAccessView(0, gpu);
            list.Dispatch(1, 1, 1);
        })
    }

    /// Milliseconds the card takes for shaders to gather from `bytes` of
    /// the buffer at GPU address `gpu` (raw view, unordered access): 64K
    /// threads, each 256 dependent 4-byte loads at pseudo-random places,
    /// the best of three. What vertex fetch and structured reads of a
    /// big title buffer do, unlike a copy's streaming.
    #[cfg(test)]
    fn gather_ms(&mut self, gpu: u64, bytes: u64) -> windows::core::Result<f64> {
        const SOURCE: &str = "#define RS \"UAV(u0), RootConstants(num32BitConstants=1, b0)\"
RWByteAddressBuffer b : register(u0);
cbuffer c : register(b0) { uint mask; };
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { uint x = id.x * 2654435761u; uint sum = 0; for (uint i = 0; i < 256; i++) { x = x * 1664525u + 1013904223u + sum; sum += b.Load((x & mask) & ~3u); } if (sum == 0x12345678) b.Store(0, sum); }
";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: as in `shade`.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("gather"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)?;
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        let mask = (bytes.next_power_of_two() >> 1).max(4) as u32 - 1;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            // SAFETY: an open list of ours; the view names a live buffer of
            // `bytes` or more, and every load lies below `mask`.
            self.run(|list| unsafe {
                list.SetComputeRootSignature(&root);
                list.SetPipelineState(&pso);
                list.SetComputeRootUnorderedAccessView(0, gpu);
                list.SetComputeRoot32BitConstant(1, mask, 0);
                list.Dispatch(1024, 1, 1);
            })?;
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(best)
    }

    /// Milliseconds the card takes to copy `bytes` of `from` into its own
    /// memory, the best of three.
    fn copy_ms(&mut self, from: &ID3D12Resource, bytes: u64) -> windows::core::Result<f64> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut to: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut to)?
        };
        let to = to.ok_or_else(windows::core::Error::empty)?;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            // SAFETY: two live buffers of `bytes` bytes or more.
            self.run(|list| unsafe { list.CopyBufferRegion(&to, 0, from, 0, bytes) })?;
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(best)
    }
}

fn fill(at: *mut u8, len: usize, seed: u8) {
    for i in 0..len {
        // SAFETY: the caller's own committed memory of `len` bytes.
        unsafe { at.add(i).write((i as u8).wrapping_mul(31).wrapping_add(seed)) };
    }
}

fn same(at: *const u8, bytes: &[u8]) -> bool {
    // SAFETY: the caller's own committed memory of that length.
    unsafe { std::slice::from_raw_parts(at, bytes.len()) == bytes }
}

/// The process's working set in MB.
fn working_set_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    let mut counters = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut counters, counters.cb) };
    counters.WorkingSetSize as u64 >> 20
}

/// What the probe found. The lines are the whole account; the flags are
/// what the stages build on.
#[derive(Default)]
pub(crate) struct Probe {
    pub(crate) lines: Vec<String>,
    /// A part of private, write-watched memory opens, a buffer in it shows
    /// the memory's bytes to the GPU, also those the CPU wrote afterwards.
    pub(crate) part_reads: bool,
    /// What the GPU writes there (a copy, and a shader through unordered
    /// access) the CPU reads at the memory's address.
    pub(crate) part_writes: bool,
    /// The same two for a committed section, through a view of it.
    pub(crate) section_reads: bool,
    pub(crate) section_writes: bool,
    /// The section's heap outlives the view: the GPU's write with no view
    /// mapped shows in a view mapped afterwards.
    pub(crate) section_outlives_views: bool,
    /// A section made with `SEC_RESERVE` (as `xmem` makes the page pool)
    /// was refused.
    pub(crate) reserved_section_refused: bool,
}

/// Tries the openings on memory of the probe's own, made the way `xmem`
/// makes title memory, with a queue of its own on `device`; `memory`
/// gives the card's memory account (`Host::video_memory`) where there is
/// one. Nothing of the title's is touched. `mb` is the size of the cost
/// measurement (0 leaves it out).
pub(crate) fn probe(device: &ID3D12Device, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) -> Probe {
    let mut p = Probe::default();
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => {
            p.lines.push(format!("no queue of its own: {e}"));
            return p;
        }
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a plain query that fills our structure.
    let _ = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32) };
    p.lines.push(format!(
        "the device: ID3D12Device3 {}, ID3D12Device13 {} (parts of an allocation); cross-adapter row-major textures {}; heap tier {}",
        device.cast::<ID3D12Device3>().is_ok(),
        device.cast::<ID3D12Device13>().is_ok(),
        options.CrossAdapterRowMajorTextureSupported.as_bool(),
        options.ResourceHeapTier.0
    ));
    // SAFETY: memory of the probe's own, made and released here; every
    // address used lies inside it.
    unsafe {
        private(&mut p, &mut rig);
        section(&mut p, &mut rig);
        if mb > 0 {
            cost(&mut p, &mut rig, memory, mb);
        }
    }
    p.lines.push(format!(
        "VERDICT: private memory in place: reads {}, writes {}; the page pool's section in place: reads {}, writes {}, whatever the views do {}; \
         a reserved section is refused {} (the pool must be made committed to be opened)",
        p.part_reads, p.part_writes, p.section_reads, p.section_writes, p.section_outlives_views, p.reserved_section_refused
    ));
    p
}

fn describe(heap: &ID3D12Heap) -> String {
    // SAFETY: a plain query.
    let d = unsafe { heap.GetDesc() };
    format!(
        "heap of {} KiB: type {} (4 = custom), CPU pages {} (3 = write-back), pool {} (1 = the PC's memory), alignment {}, flags {:#x} (0x1 shared, 0x20 cross-adapter, 0x200 write watch, 0x400 shader atomics)",
        d.SizeInBytes >> 10,
        d.Properties.Type.0,
        d.Properties.CPUPageProperty.0,
        d.Properties.MemoryPoolPreference.0,
        d.Alignment,
        d.Flags.0
    )
}

/// Private memory, reserved as `xmem` reserves the title's heaps
/// (committed, read-write, write-watched): a part of it opened.
unsafe fn private(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 64 * K64;
    // SAFETY (the whole function): the allocation below, `SIZE` bytes.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if base.is_null() {
            p.lines.push("private memory: none to try with".to_owned());
            return;
        }
        // The part: 16 pages from the 16th on.
        let (at, bytes) = (base.add(16 * K64), 16 * K64);
        let whole = open(&rig.device, From::Region(base as usize));
        p.lines.push(format!("private memory, the whole allocation (OpenExistingHeapFromAddress): {}", hr(&whole)));
        drop(whole);
        let opened = open(&rig.device, From::Part(at as usize, bytes));
        p.lines.push(format!("private memory, a part of it (OpenExistingHeapFromAddress1): {}", hr(&opened)));
        if let Ok(heap) = &opened {
            p.lines.push(format!("  {}", describe(heap)));
            let placed = place(&rig.device, heap, K64 as u64, 4 * K64 as u64);
            p.lines.push(format!("  a buffer placed in it (cross-adapter, unordered access): {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = at.add(K64);
                fill(memory, 4 * K64, 1);
                let first = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                fill(memory, 4 * K64, 77);
                let second = rig.fetch(buffer, 0, 4 * K64).is_ok_and(|got| same(memory, &got));
                p.part_reads = first && second;
                p.lines.push(format!("  the GPU reads the memory's bytes: {first}; and those the CPU wrote afterwards, with no call in between: {second}"));
                mem::ResetWriteWatch(base.cast(), SIZE);
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x5a).collect();
                let copied = rig.store(buffer, 4096, &new).is_ok() && same(memory.add(4096), &new);
                let (mut found, mut count, mut granularity) = ([0usize; 64], 64usize, 0u32);
                let watched = mem::GetWriteWatch(0, base.cast(), SIZE, found.as_mut_ptr().cast(), &mut count, &mut granularity) == 0;
                memory.cast::<[u32; 3]>().write([40, 2, 0]);
                let shaded = rig.shade(buffer.GetGPUVirtualAddress());
                let shader = shaded.is_ok() && memory.cast::<[u32; 3]>().read()[2] == 42;
                p.part_writes = copied && shader;
                p.lines.push(format!(
                    "  the CPU reads at the memory's address what the GPU wrote: by a copy {copied}; by a shader through unordered access {shader} ({}); \
                     write watching saw {} of the copy's 16 pages{}",
                    hr(&shaded),
                    if watched { count } else { 0 },
                    if watched { "" } else { " (the asking failed)" }
                ));
            }
            // What the title can still do with the pages while the heap stands.
            let mut old = 0;
            let protect_in = mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
            if protect_in {
                mem::VirtualProtect(at.add(8 * K64).cast(), K64, mem::PAGE_READWRITE, &mut old);
            }
            let decommit_in = mem::VirtualFree(at.add(9 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            let decommit_out = mem::VirtualFree(base.add(40 * K64).cast(), K64, mem::MEM_DECOMMIT) != 0;
            p.lines.push(format!(
                "  with the heap standing, the title's own calls: read-only inside the part {protect_in}, decommit inside {decommit_in}, decommit outside {decommit_out} \
                 (false inside means the heap must go before the title frees or guards those pages)"
            ));
            drop(placed);
        }
        drop(opened);
        let freed = mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE) != 0;
        p.lines.push(format!("  released after the heap went: {freed}"));
    }
}

/// A section and a view of it in a placeholder, as `xmem` maps physical
/// pages: reserved (as it is made today) and committed.
unsafe fn section(p: &mut Probe, rig: &mut Rig) {
    const SIZE: usize = 32 * K64;
    // SAFETY (the whole function): the sections, placeholder and views
    // made below.
    unsafe {
        let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let reserved = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_RESERVE, 0, SIZE as u32, std::ptr::null());
        if !reserved.is_null() {
            let opened = open(&rig.device, From::Section(reserved as usize));
            p.reserved_section_refused = opened.is_err();
            p.lines.push(format!("a section made reserved, as the page pool is today (OpenExistingHeapFromFileMapping): {}", hr(&opened)));
            drop(opened);
            windows_sys::Win32::Foundation::CloseHandle(reserved);
        }
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, SIZE as u32, std::ptr::null());
        let hole = mem::VirtualAlloc2(process, std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0).cast::<u8>();
        if handle.is_null() || hole.is_null() {
            p.lines.push("a committed section: none to try with".to_owned());
            return;
        }
        // Pages 4..12 of the section at the placeholder's start.
        mem::VirtualFree(hole.cast(), 8 * K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
        let view = mem::MapViewOfFile3(handle, process, hole.cast(), (4 * K64) as u64, 8 * K64, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
        let opened = open(&rig.device, From::Section(handle as usize));
        p.lines.push(format!("a section made committed (OpenExistingHeapFromFileMapping): {}", hr(&opened)));
        let mut unmapped = false;
        if let (Ok(heap), false) = (&opened, view.Value.is_null()) {
            p.lines.push(format!("  {}", describe(heap)));
            // The section's page 5 is the view's second page.
            let placed = place(&rig.device, heap, 5 * K64 as u64, 2 * K64 as u64);
            p.lines.push(format!("  a buffer placed at the section's offset of a view's pages: {}", hr(&placed)));
            if let Ok(buffer) = &placed {
                let memory = hole.add(K64);
                fill(memory, 2 * K64, 3);
                p.section_reads = rig.fetch(buffer, 0, 2 * K64).is_ok_and(|got| same(memory, &got));
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x33).collect();
                p.section_writes = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The guard of `watch.rs`: the view read-only.
                let mut old = 0;
                let guarded = mem::VirtualProtect(memory.cast(), K64, mem::PAGE_READONLY, &mut old) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x44).collect();
                let through = rig.store(buffer, 0, &new).is_ok() && same(memory, &new);
                // The view goes; the GPU writes; another view shows it.
                unmapped = mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0;
                let new: Vec<u8> = (0..K64).map(|i| (i as u8) ^ 0x55).collect();
                let stored = rig.store(buffer, 0, &new).is_ok();
                let other = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, (5 * K64) as u32, K64);
                p.section_outlives_views = unmapped && stored && !other.Value.is_null() && same(other.Value.cast(), &new);
                p.lines.push(format!(
                    "  the GPU reads the view's bytes {}; the CPU reads through the view what the GPU wrote {}; the view made read-only under the heap {guarded}, \
                     and the GPU writes it all the same {through}; the view unmapped under the heap {unmapped}, the GPU's write then shows in a view mapped afterwards {}",
                    p.section_reads, p.section_writes, p.section_outlives_views
                ));
                if !other.Value.is_null() {
                    mem::UnmapViewOfFile(other);
                }
            }
        }
        drop(opened);
        if !unmapped && !view.Value.is_null() {
            mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER);
        }
        windows_sys::Win32::Foundation::CloseHandle(handle);
        // The placeholder was split in two: each half is released.
        mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
        mem::VirtualFree(hole.add(8 * K64).cast(), 0, mem::MEM_RELEASE);
    }
}

/// What opening costs, on `mb` MB each of private memory never touched
/// and of a committed section never touched: the time, the working set,
/// the card's account of the PC's memory shared with it; and how fast the
/// card copies out of such a heap against a shared buffer of today's kind.
unsafe fn cost(p: &mut Probe, rig: &mut Rig, memory: &dyn Fn() -> Option<[u64; 4]>, mb: usize) {
    let bytes = mb << 20;
    let shared = || memory().map_or(0, |m| m[3] >> 20);
    let ms = |speed: Option<f64>| speed.map_or("?".to_owned(), |ms| format!("{ms:.1}"));
    // SAFETY (the whole function): the allocation and the section below.
    unsafe {
        let base = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
        if !base.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Part(base as usize, bytes));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of private memory never touched: {}; opened in {took:.1} ms; working set {set} -> {} MB; shared with the card {account} -> {} MB; \
                 the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
        let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, (bytes as u64 >> 32) as u32, bytes as u32, std::ptr::null());
        if !handle.is_null() {
            let (set, account, started) = (working_set_mb(), shared(), std::time::Instant::now());
            let opened = open(&rig.device, From::Section(handle as usize));
            let took = started.elapsed().as_secs_f64() * 1000.0;
            let (set_opened, account_opened) = (working_set_mb(), shared());
            let placed = opened.as_ref().ok().and_then(|heap| place(&rig.device, heap, 0, bytes as u64).ok());
            let speed = placed.as_ref().and_then(|buffer| rig.copy_ms(buffer, bytes as u64).ok());
            p.lines.push(format!(
                "cost, {mb} MB of a committed section never touched: {}; opened in {took:.1} ms; working set {set} -> {set_opened} -> after the card read all of it {} MB; \
                 shared with the card {account} -> {account_opened} -> {} MB; the card copies it out in {} ms",
                hr(&opened),
                working_set_mb(),
                shared(),
                ms(speed)
            ));
            drop((placed, opened));
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
        // Today's kind of host copy, for the comparison.
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let made = rig.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        );
        if let (Ok(()), Some(buffer)) = (made, buffer) {
            let speed = rig.copy_ms(&buffer, bytes as u64).ok();
            p.lines.push(format!("cost, for comparison: the card copies {mb} MB out of a shared buffer of today's kind in {} ms", ms(speed)));
        }
    }
}

// The census: which of the title memory that has host copies today could
// be opened, and by which opening.

/// Counts and bytes by kind, in the order of [`KINDS`].
#[derive(Default, Clone, Copy)]
struct Tally([(u64, u64); 6]);

const KINDS: [&str; 6] = [
    "private, committed whole (a part can be opened)",
    "view memory, one run of a section (one placed buffer)",
    "view memory, scattered pages",
    "view memory with unmapped pages",
    "private but not committed whole, or guarded",
    "other",
];

fn kind_of(at: usize, len: usize, known: &mut Vec<(usize, usize)>) -> usize {
    match cover(at, len) {
        Cover::Run(..) => return 1,
        Cover::Scattered => return 2,
        Cover::Holes => return 3,
        Cover::None => {}
    }
    let end = at + len;
    if known.iter().any(|&(from, to)| from <= at && end <= to) {
        return 0;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    if n == 0 {
        return 5;
    }
    let reach = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    if info.Type == mem::MEM_PRIVATE && info.State == mem::MEM_COMMIT && info.Protect == mem::PAGE_READWRITE {
        known.push((info.BaseAddress as usize, reach));
        return if reach >= end { 0 } else { 4 };
    }
    match info.Type {
        mem::MEM_PRIVATE => 4,
        mem::MEM_MAPPED => 3,
        _ => 5,
    }
}

/// The census lines for `spans` (start, bytes, whether a window outside
/// buffers).
fn census_of(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let mut known = Vec::new();
    let (mut buffers, mut windows) = (Tally::default(), Tally::default());
    for &(start, len, window) in spans {
        let kind = kind_of(start as usize, len as usize, &mut known);
        let tally = if window { &mut windows } else { &mut buffers };
        tally.0[kind].0 += 1;
        tally.0[kind].1 += len;
    }
    let mut lines = Vec::new();
    for (name, tally) in [("title buffers", buffers), ("windows outside buffers", windows)] {
        let all = tally.0.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        lines.push(format!("census, {name}: {} of {} MB", all.0, all.1 >> 20));
        for (kind, &(n, bytes)) in KINDS.iter().zip(&tally.0) {
            if n > 0 {
                lines.push(format!("census,   {kind}: {n} of {} MB", bytes >> 20));
            }
        }
    }
    let v = views();
    let (count, bytes) = (v.views.len(), v.views.values().map(|view| view.0 as u64).sum::<u64>());
    let pages = v.views.values().filter(|view| view.0 == K64).count();
    let big = v.views.values().map(|view| view.0).max().unwrap_or(0);
    lines.push(format!(
        "census, the title's views of physical pages: {count} of {} MB; {pages} of them one 64 KiB page; the longest {} MB",
        bytes >> 20,
        big >> 20
    ));
    for &(_, bytes, committed) in &v.sections {
        lines.push(format!(
            "census, a section of {} MB, {}",
            bytes >> 20,
            if committed { "committed: it can be opened as one heap" } else { "reserved: refused as a heap until it is made with SEC_COMMIT" }
        ));
    }
    lines
}

/// With [`probe_level`] 2: one part of the title's own private memory
/// (the start of the biggest such title buffer, 4 MiB at most) is opened
/// for a moment, a buffer placed, 64 KiB read by the GPU and compared
/// with what the CPU reads there (the title may be writing it meanwhile:
/// the count of equal bytes is given), and all of it released. While the
/// heap stands (milliseconds) a `VirtualFree` of the title's on those
/// pages would fail.
fn real_part(spans: &[(u64, u64, bool)]) -> Vec<String> {
    let Some(host) = crate::host::host() else {
        return Vec::new();
    };
    let mut known = Vec::new();
    let found = spans
        .iter()
        .filter(|s| !s.2 && s.1 >= 2 * K64 as u64)
        .filter(|s| kind_of(s.0 as usize, s.1 as usize, &mut known) == 0)
        .max_by_key(|s| s.1);
    let Some(&(start, len, _)) = found else {
        return vec!["the title's memory: no title buffer in private, committed memory to try".to_owned()];
    };
    let at = (start as usize).next_multiple_of(K64);
    let bytes = ((start + len) as usize).saturating_sub(at).min(4 << 20) & !(K64 - 1);
    if bytes == 0 || !private_whole(at, bytes) {
        return vec!["the title's memory: the buffer found holds no whole 64 KiB page that is still committed".to_owned()];
    }
    let mut lines = Vec::new();
    let (set, started) = (working_set_mb(), std::time::Instant::now());
    let opened = open(&host.device, From::Part(at, bytes));
    lines.push(format!(
        "the title's memory, {} KiB at {at:#x} (of the title buffer {start:#x}+{len:#x}): {}; in {:.1} ms; working set {set} -> {} MB",
        bytes >> 10,
        hr(&opened),
        started.elapsed().as_secs_f64() * 1000.0,
        working_set_mb()
    ));
    if let Ok(heap) = &opened {
        let placed = place(&host.device, heap, 0, bytes as u64);
        lines.push(format!("  a buffer placed over it: {}", hr(&placed)));
        if let (Ok(buffer), Ok(mut rig)) = (&placed, Rig::new(&host.device)) {
            let mut before = vec![0u8; K64];
            crate::host::copy_mapped(before.as_mut_ptr(), at as *const u8, K64, false);
            let got = rig.fetch(buffer, 0, K64);
            let mut after = vec![0u8; K64];
            crate::host::copy_mapped(after.as_mut_ptr(), at as *const u8, K64, false);
            match got {
                Ok(got) => {
                    let equal = (0..K64).filter(|&i| got[i] == before[i] || got[i] == after[i]).count();
                    let zeros = got.iter().filter(|&&b| b == 0).count();
                    lines.push(format!(
                        "  the GPU read 64 KiB of the title's own bytes: {equal} of {K64} equal to what the CPU read just before or just after ({zeros} of them zero)"
                    ));
                }
                Err(e) => lines.push(format!("  the GPU's read failed: {e}")),
            }
        }
    }
    lines
}

/// Called once a second by the thread that submits, before a submission
/// (`host::sweep`): with [`probe_level`], the probe at the first call, the
/// census (on a thread of its own: it asks Windows about every copy) 30
/// and 150 seconds later.
pub(crate) fn tick() {
    if level() > 0 {
        say();
    }
    if probe_level() == 0 {
        return;
    }
    static STARTED: AtomicU64 = AtomicU64::new(0);
    static DONE: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms().max(1);
    let started = match STARTED.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => {
            let Some(host) = crate::host::host() else {
                return;
            };
            let mb = std::env::var("VICTORY_INPLACE_PROBE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64usize);
            let took = std::time::Instant::now();
            let probe = probe(&host.device, &|| host.video_memory(), mb.min(1024));
            for line in &probe.lines {
                tracing::info!("in place probe: {line}");
            }
            tracing::info!("in place probe: took {:.0} ms", took.elapsed().as_secs_f64() * 1000.0);
            now
        }
        Err(started) => started,
    };
    let due = match DONE.load(Ordering::Relaxed) {
        0 => now >= started + 30_000,
        1 => now >= started + 150_000,
        _ => false,
    };
    if due {
        let which = DONE.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            let spans = crate::shadow::spans();
            for line in census_of(&spans) {
                tracing::info!("in place {line}");
            }
            if which == 0 && probe_level() >= 2 {
                for line in real_part(&spans) {
                    tracing::info!("in place probe: {line}");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Views as the runtime tells them: a range inside one view, or over
    /// views that continue each other in one section, is one run; over
    /// views from elsewhere it is scattered; over a gap it has holes.
    #[test]
    fn views_say_what_one_placed_buffer_could_show() {
        let mut views = BTreeMap::new();
        let at = 0x7_0000_0000usize;
        views.insert(at, (4 * K64, 9, 16 * K64 as u64));
        // Continues the first in the section.
        views.insert(at + 4 * K64, (2 * K64, 9, 20 * K64 as u64));
        // Follows in address, from elsewhere in the section.
        views.insert(at + 6 * K64, (K64, 9, 3 * K64 as u64));
        // After a gap of one page.
        views.insert(at + 8 * K64, (K64, 9, 4 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 2 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at + K64, 5 * K64), Cover::Run(9, 17 * K64 as u64));
        assert_eq!(cover_in(&views, at, 7 * K64), Cover::Scattered);
        assert_eq!(cover_in(&views, at + 6 * K64, 3 * K64), Cover::Holes);
        assert_eq!(cover_in(&views, at + 7 * K64, K64), Cover::None);
        assert_eq!(cover_in(&views, at - K64, 2 * K64), Cover::None);
    }

    /// A range whose section pages another view shows too has aliases,
    /// where those pages are; one whose pages only it shows, or that no
    /// view holds, has none.
    #[test]
    fn pages_shown_at_two_addresses_are_aliased() {
        let mut views = Views::new();
        let (a, b) = (0x7_0000_0000usize, 0x8_0000_0000usize);
        views.insert(a, 4 * K64, 9, 0);
        // Pages 2 and 3 of section 9 again, elsewhere.
        views.insert(b, 2 * K64, 9, 2 * K64 as u64);
        // Another section at the same offsets.
        views.insert(b + 4 * K64, 4 * K64, 5, 0);
        assert!(aliases_in(&views, a, 2 * K64).is_empty());
        assert_eq!(aliases_in(&views, a + K64, 2 * K64), [(a + 2 * K64, b, K64)]);
        assert_eq!(aliases_in(&views, b, K64), [(b, a + 2 * K64, K64)]);
        assert!(aliases_in(&views, b + 4 * K64, 4 * K64).is_empty());
        assert!(aliases_in(&views, a - 2 * K64, K64).is_empty());
        // A view longer than `BIG` (a pool view), kept apart, and one page
        // of it shown elsewhere too: found from either side.
        let (c, d) = (0x9_0000_0000usize, 0xa_0000_0000usize);
        let mid = 20 << 20;
        views.insert(c, 2 * BIG, 3, 0);
        views.insert(d, K64, 3, mid as u64);
        assert_eq!(aliases_in(&views, d, K64), [(d, c + mid, K64)]);
        assert_eq!(aliases_in(&views, c + mid, K64), [(c + mid, d, K64)]);
        assert!(aliases_in(&views, c, K64).is_empty());
    }

    /// What the runtime forgets when memory is mapped anew.
    #[test]
    fn remapped_memory_forgets_its_views() {
        let at = 0x7_4000_0000usize;
        title_memory_view(at, 4 * K64, 77, 0);
        title_memory_view(at + 4 * K64, K64, 77, 4 * K64 as u64);
        assert_eq!(cover(at, 5 * K64), Cover::Run(77, 0));
        // One page inside the first view goes: the whole view is forgotten
        // (the runtime maps and tells again what stays of it).
        views_gone(at + K64, K64);
        assert_eq!(cover(at, K64), Cover::None);
        assert_eq!(cover(at + 4 * K64, K64), Cover::Run(77, 4 * K64 as u64));
        views_gone(at + 4 * K64, K64);
        assert_eq!(cover(at + 4 * K64, K64), Cover::None);
    }

    /// Only private memory committed whole, read-write, is taken for a part.
    #[test]
    fn only_private_memory_committed_whole_is_a_part() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert!(!private_whole(base as usize, K64), "reserved only");
            mem::VirtualAlloc(base.cast(), 4 * K64, mem::MEM_COMMIT, mem::PAGE_READWRITE);
            assert!(private_whole(base as usize, 4 * K64));
            assert!(private_whole(base as usize + K64, 2 * K64));
            assert!(!private_whole(base as usize, 5 * K64), "reaches into the reserved half");
            let mut old = 0;
            mem::VirtualProtect(base.cast(), K64, mem::PAGE_READONLY, &mut old);
            assert!(!private_whole(base as usize, 2 * K64), "a guarded page");
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    pub(crate) fn warp() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation on the software
        // adapter; a card is never opened.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                return None;
            }
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            device
        }
    }

    /// The first hardware adapter, for the probes that measure the card
    /// itself (ignored tests: `cargo test -p victory-gpu -- --ignored card_`).
    pub(crate) fn card() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI queries and device creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            for i in 0.. {
                let adapter: IDXGIAdapter1 = factory.EnumAdapters1(i).ok()?;
                let desc = adapter.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_ok() {
                    let name = String::from_utf16_lossy(&desc.Description);
                    eprintln!("card: {}", name.trim_end_matches('\0'));
                    return device;
                }
            }
            None
        }
    }

    /// The probe on the card, with the cost measured over 512 MB: what the
    /// card's driver makes of title memory in place (the moonshot's first
    /// phase, MOONSHOT.md).
    #[test]
    #[ignore = "opens the card"]
    fn card_probe_opens_title_like_memory_and_the_gpu_sees_its_bytes() {
        let Some(device) = card() else { return };
        let probe = probe(&device, &|| None, 512);
        for line in &probe.lines {
            eprintln!("in place probe: {line}");
        }
        assert!(probe.section_reads && probe.section_writes, "a committed section in place");
    }

    /// The probe, whole, on the software adapter: the same calls the card
    /// gets. The runtime's and the kernel's rules hold there; the card's
    /// driver is what a run with `VICTORY_INPLACE_PROBE=1` adds.
    #[test]
    fn warp_probe_opens_title_like_memory_and_the_gpu_sees_its_bytes() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            eprintln!("no ID3D12Device13 here: parts of an allocation cannot be opened");
            return;
        }
        let probe = probe(&device, &|| None, 4);
        for line in &probe.lines {
            eprintln!("in place probe: {line}");
        }
        assert!(probe.part_reads, "the GPU reads private memory in place");
        assert!(probe.part_writes, "GPU writes land in private memory");
        assert!(probe.section_reads && probe.section_writes, "a committed section in place");
        assert!(probe.section_outlives_views, "the section's heap does not depend on views");
        assert!(probe.reserved_section_refused, "a reserved section is refused");
    }

    /// Stage 1's own call on the software adapter: a title buffer that
    /// begins and ends off the 64 KiB lines is placed over its pages, the
    /// GPU reads the buffer's bytes at the offset the views use, and its
    /// write lands at the title's address. Memory that is not committed
    /// whole is turned down without asking the host.
    #[test]
    fn warp_a_title_buffer_in_place_shows_its_bytes_at_its_offset() {
        let Some(device) = warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE).cast::<u8>();
            let (start, len) = (base as u64 + K64 as u64 + 0x1230, 2 * K64 as u64 + 5);
            let placed = place_part(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x1230);
            assert_eq!(placed.gpu, placed.buffer.GetGPUVirtualAddress() + 0x1230);
            assert_eq!(placed.buffer.GetDesc().Width, 3 * K64 as u64, "the pages that hold the buffer");
            let mut rig = Rig::new(&device).unwrap();
            fill(start as *mut u8, len as usize, 7);
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(same(start as *const u8, &got), "the GPU reads the title buffer's bytes");
            rig.store(&placed.buffer, placed.bias + 16, &[0xa5; 64]).unwrap();
            assert!(same((start + 16) as *const u8, &[0xa5; 64]), "the GPU's write is at the title's address");
            drop(placed);
            // Half of it decommitted: turned down, the host not asked.
            assert_ne!(mem::VirtualFree(base.add(2 * K64).cast(), K64, mem::MEM_DECOMMIT), 0);
            assert!(matches!(place_part(&device, start, len), Err(None)));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    /// View memory in place (MOONSHOT.md, phase 2): pages 4..12 of a
    /// committed section shown at a placeholder, the runtime's calls made
    /// as `xmem` makes them, then a title buffer inside the view placed in
    /// the section's heap. The GPU reads the view's bytes at the buffer's
    /// offset and its write shows at the title's address; the heap is
    /// opened once for both buffers placed; a view of a reserved section is
    /// turned down without asking the host.
    fn view_memory_goes_in_place(device: &ID3D12Device) {
        // SAFETY: the sections, placeholder and views of the test's own.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            let size = 16 * K64;
            let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, size as u32, std::ptr::null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), size, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0).cast::<u8>();
            assert!(!handle.is_null() && !hole.is_null());
            mem::VirtualFree(hole.cast(), 8 * K64, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
            let view = mem::MapViewOfFile3(handle, process, hole.cast(), (4 * K64) as u64, 8 * K64, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert_eq!(view.Value.cast::<u8>(), hole);
            title_memory_section(handle as usize, size as u64, true);
            title_memory_view(hole as usize, 8 * K64, handle as usize, (4 * K64) as u64);
            // A buffer from 0x1230 into the view's second page, 2 pages and 5 bytes long.
            let (start, len) = (hole as u64 + K64 as u64 + 0x1230, 2 * K64 as u64 + 5);
            let Cover::Run(section, offset) = cover(start as usize, len as usize) else { panic!("one run of one section") };
            assert_eq!((section, offset), (handle as usize, 5 * K64 as u64 + 0x1230));
            let placed = place_view(device, section, offset, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert!(!placed.pins);
            assert_eq!(placed.bias, 0x1230);
            assert_eq!(placed.buffer.GetDesc().Width, 3 * K64 as u64, "the pages that hold the buffer");
            let mut rig = Rig::new(device).unwrap();
            fill(start as *mut u8, len as usize, 9);
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(same(start as *const u8, &got), "the GPU reads the view's bytes");
            rig.store(&placed.buffer, placed.bias + 16, &[0x5a; 64]).unwrap();
            assert!(same((start + 16) as *const u8, &[0x5a; 64]), "the GPU's write is at the title's address");
            let other = place_view(device, section, 4 * K64 as u64, hole as u64, 64).unwrap();
            assert!(other.heap == placed.heap, "one opening per section");
            drop((placed, other));
            // The same handle named again for a reserved section: the
            // heap is forgotten, and its views are turned down.
            title_memory_section(handle as usize, size as u64, false);
            assert!(matches!(place_view(device, section, offset, start, len), Err(None)));
            views_gone(hole as usize, 8 * K64);
            mem::UnmapViewOfFile2(process, view, 0);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
    }

    /// MOONSHOT.md, phase 3b: the pool as heaps the card makes, not heaps
    /// opened on our sections. The title maps its physical pages out of
    /// order, so a buffer of view memory is seldom one run of a section; the
    /// console's answer is the GPU's page tables, whose host form is a
    /// reserved resource with tile mappings, which an opened heap must never
    /// take (it removes the device). A heap the card made may take them; the
    /// question is whether the CPU can show its pages at addresses of ours,
    /// through the heap's shared handle mapped as a section. Each step says
    /// what it found; nothing is asserted but that the steps ran.
    fn card_heap_as_the_pool(device: &ID3D12Device) -> Vec<String> {
        const TILE: usize = 64 << 10;
        const TILES: usize = 16;
        let mut lines = Vec::new();
        let mut rig = match Rig::new(device) {
            Ok(rig) => rig,
            Err(e) => return vec![format!("no rig: {e}")],
        };
        let wb = (D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0);
        let wc = (D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0);
        let na = (D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0);
        let shared = D3D12_HEAP_FLAG_SHARED;
        let cross = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
        let buffers = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        for (name, (page, pool), flags) in [
            ("write-back, shared", wb, shared),
            ("write-back, shared, buffers", wb, shared | buffers),
            ("write-back, cross-adapter", wb, cross),
            ("write-combine, shared", wc, shared),
            ("write-combine, cross-adapter", wc, cross),
            ("not CPU-visible in the PC's memory, cross-adapter", na, cross),
            ("not CPU-visible in the PC's memory, shared", na, shared),
        ] {
            // SAFETY (the loop body): objects and views of the test's own;
            // every address used lies in a view mapped here.
            unsafe {
                let desc = D3D12_HEAP_DESC {
                    SizeInBytes: (TILES * TILE) as u64,
                    Properties: D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_CUSTOM,
                        CPUPageProperty: page,
                        MemoryPoolPreference: pool,
                        ..Default::default()
                    },
                    Alignment: 0,
                    Flags: flags,
                };
                let mut heap: Option<ID3D12Heap> = None;
                if let Err(e) = device.CreateHeap(&desc, &mut heap) {
                    lines.push(format!("{name}: the card makes no such heap: {e}"));
                    continue;
                }
                let heap = heap.unwrap();
                let handle = match device.CreateSharedHandle(&heap, None, 0x1000_0000, windows::core::PCWSTR::null()) {
                    Ok(h) => h,
                    Err(e) => {
                        lines.push(format!("{name}: heap made, no shared handle: {e}"));
                        continue;
                    }
                };
                let view = mem::MapViewOfFile(handle.0, mem::FILE_MAP_ALL_ACCESS, 0, 0, TILES * TILE);
                if view.Value.is_null() {
                    lines.push(format!(
                        "{name}: heap made, shared handle made, the CPU cannot map it as a section: {}",
                        std::io::Error::last_os_error()
                    ));
                    windows_sys::Win32::Foundation::CloseHandle(handle.0);
                    continue;
                }
                let cpu = view.Value.cast::<u8>();
                for t in 0..TILES {
                    std::ptr::write_bytes(cpu.add(t * TILE), t as u8 + 1, TILE);
                }
                lines.push(format!("{name}: the CPU maps the card's heap as a section, read-write"));
                // The page table: tile k of a reserved buffer shows heap tile 15 - k.
                let mut reserved: Option<ID3D12Resource> = None;
                let made = device.CreateReservedResource(
                    &buffer_desc((TILES * TILE) as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut reserved,
                );
                match (made, reserved) {
                    (Ok(()), Some(reserved)) => {
                        let starts: Vec<D3D12_TILED_RESOURCE_COORDINATE> =
                            (0..TILES as u32).map(|k| D3D12_TILED_RESOURCE_COORDINATE { X: k, Y: 0, Z: 0, Subresource: 0 }).collect();
                        let sizes = vec![D3D12_TILE_REGION_SIZE { NumTiles: 1, UseBox: false.into(), Width: 1, Height: 1, Depth: 1 }; TILES];
                        let flags = [D3D12_TILE_RANGE_FLAG_NONE; TILES];
                        let offsets: Vec<u32> = (0..TILES as u32).map(|k| TILES as u32 - 1 - k).collect();
                        let counts = [1u32; TILES];
                        rig.queue.UpdateTileMappings(
                            &reserved,
                            TILES as u32,
                            Some(starts.as_ptr()),
                            Some(sizes.as_ptr()),
                            &heap,
                            TILES as u32,
                            Some(flags.as_ptr()),
                            Some(offsets.as_ptr()),
                            Some(counts.as_ptr()),
                            D3D12_TILE_MAPPING_FLAG_NONE,
                        );
                        let read = rig.fetch(&reserved, 0, TILES * TILE);
                        let ordered = read.as_ref().is_ok_and(|got| (0..TILES).all(|k| got[k * TILE] == (TILES - k) as u8 && got[k * TILE + TILE - 1] == (TILES - k) as u8));
                        lines.push(format!(
                            "{name}: a reserved buffer tile-mapped onto it in reverse: {}",
                            match &read {
                                Ok(_) if ordered => "the GPU reads each tile's CPU bytes through the page table".to_owned(),
                                Ok(got) => format!("read, but not the CPU's bytes (first byte {})", got[0]),
                                Err(e) => format!("not read: {e}"),
                            }
                        ));
                        let wrote = rig.store(&reserved, 0, &[0xc3; 256]).is_ok() && same(cpu.add((TILES - 1) * TILE), &[0xc3; 256]);
                        lines.push(format!("{name}: the GPU's write through tile 0 shows at the CPU's tile 15: {wrote}"));
                        let removed = device.GetDeviceRemovedReason();
                        lines.push(format!("{name}: the device after the tile mappings: {}", if removed.is_ok() { "fine".to_owned() } else { format!("{removed:?}") }));
                    }
                    (made, _) => lines.push(format!("{name}: no reserved buffer: {made:?}")),
                }
                mem::UnmapViewOfFile(view);
                windows_sys::Win32::Foundation::CloseHandle(handle.0);
            }
        }
        lines
    }

    /// Why world run 2 spent twice the card's time with buffers in place:
    /// scattered shader reads from memory the card made against memory
    /// opened in place (private and a section), 256 MB each.
    #[test]
    #[ignore = "opens the card"]
    fn card_gathers_from_memory_in_place() {
        let Some(device) = card() else { return };
        let mut rig = Rig::new(&device).unwrap();
        let bytes = 256usize << 20;
        let mut line = |name: &str, buffer: &ID3D12Resource| {
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let round = |r: windows::core::Result<f64>| r.map(|m| (m * 10.0).round() / 10.0);
            let gather = round(rig.gather_ms(gpu, bytes as u64));
            // The same reads over 1 MiB, which the card's cache holds.
            let hot = round(rig.gather_ms(gpu, 1 << 20));
            let copy = round(rig.copy_ms(buffer, bytes as u64));
            eprintln!("gather probe: {name}: scattered reads over 256 MB {gather:?} ms, over 1 MB {hot:?} ms, a copy {copy:?} ms");
        };
        // SAFETY: objects and memory of the test's own.
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut made)
                .unwrap();
            line("a host copy of today's kind (committed, the PC's memory)", &made.unwrap());
            let card = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut made: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&card, D3D12_HEAP_FLAG_NONE, &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut made)
                .unwrap();
            line("the card's own memory", &made.unwrap());
            let base = mem::VirtualAlloc(std::ptr::null(), bytes, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE);
            let placed = place_part(&device, base as u64, bytes as u64).unwrap_or_else(|e| panic!("{e:?}"));
            line("private memory in place", &placed.buffer);
            drop(placed);
            mem::VirtualFree(base, 0, mem::MEM_RELEASE);
            let invalid = windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            let handle = mem::CreateFileMappingW(invalid, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_COMMIT, 0, bytes as u32, std::ptr::null());
            let heap = open(&device, From::Section(handle as usize)).unwrap();
            let buffer = place(&device, &heap, 0, bytes as u64).unwrap();
            line("a committed section in place", &buffer);
            drop((buffer, heap));
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
    }

    /// Resizable BAR as Direct3D shows it (`D3D12_HEAP_TYPE_GPU_UPLOAD`):
    /// the card's own memory, mapped for the CPU. Whether this PC has it,
    /// and what it costs: the CPU writing and reading 256 MB of it, and
    /// the card's scattered reads from it (against its other memory in
    /// `card_gathers_from_memory_in_place`).
    #[test]
    #[ignore = "opens the card"]
    fn card_gpu_upload_heap_probe() {
        let Some(device) = card() else { return };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        // SAFETY: a plain query that fills our structure.
        let asked = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut options).cast(), size_of_val(&options) as u32) };
        eprintln!("rebar probe: options 16 asked: {asked:?}; GPU upload heaps supported: {}", options.GPUUploadHeapSupported.as_bool());
        if asked.is_err() || !options.GPUUploadHeapSupported.as_bool() {
            return;
        }
        let bytes = 256usize << 20;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: plain creation and mapping of a buffer of ours.
        unsafe {
            if let Err(e) = device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut made,
            ) {
                eprintln!("rebar probe: no such buffer: {e}");
                return;
            }
            let buffer = made.unwrap();
            let mut data = std::ptr::null_mut();
            buffer.Map(0, None, Some(&mut data)).unwrap();
            let source = vec![0x5au8; bytes];
            let started = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(source.as_ptr(), data.cast::<u8>(), bytes);
            let write = started.elapsed().as_secs_f64();
            let mut back = vec![0u8; 16 << 20];
            let started = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(data.cast::<u8>(), back.as_mut_ptr(), back.len());
            let read = started.elapsed().as_secs_f64();
            eprintln!(
                "rebar probe: the CPU writes {:.1} GB/s into the card's memory, reads {:.2} GB/s back; bytes read back right: {}",
                bytes as f64 / write / 1e9,
                back.len() as f64 / read / 1e9,
                back.iter().all(|&b| b == 0x5a)
            );
            let mut rig = Rig::new(&device).unwrap();
            let gpu = buffer.GetGPUVirtualAddress();
            eprintln!(
                "rebar probe: the card's scattered reads from it: over 256 MB {:?} ms, over 1 MB {:?} ms",
                rig.gather_ms(gpu, bytes as u64),
                rig.gather_ms(gpu, 1 << 20)
            );
        }
    }

    #[test]
    #[ignore = "opens the card"]
    fn card_heap_as_the_pool_probe() {
        let Some(device) = card() else { return };
        for line in card_heap_as_the_pool(&device) {
            eprintln!("phase 3b probe: {line}");
        }
    }

    #[test]
    fn warp_view_memory_goes_in_place() {
        let Some(device) = warp() else { return };
        view_memory_goes_in_place(&device);
    }

    #[test]
    #[ignore = "opens the card"]
    fn card_view_memory_goes_in_place() {
        let Some(device) = card() else { return };
        view_memory_goes_in_place(&device);
    }

    /// The census names each kind of memory.
    #[test]
    fn census_tells_private_memory_from_views() {
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 * K64, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as u64;
            let view = 0x7_8000_0000u64;
            title_memory_view(view as usize, 2 * K64, 5, 0);
            title_memory_view(view as usize + 2 * K64, K64, 5, 9 * K64 as u64);
            let lines = census_of(&[(base, 2 * K64 as u64, false), (base + 4 * K64 as u64, K64 as u64, true), (view, K64 as u64, true), (view, 3 * K64 as u64, false)]);
            let text = lines.join("\n");
            assert!(text.contains("census, title buffers: 2 of"), "{text}");
            assert!(text.contains("census,   private, committed whole (a part can be opened): 1 of"), "{text}");
            assert!(text.contains("census,   view memory, scattered pages: 1 of"), "{text}");
            assert!(text.contains("census,   view memory, one run of a section (one placed buffer): 1 of"), "{text}");
            views_gone(view as usize, 3 * K64);
            mem::VirtualFree(base as *mut c_void, 0, mem::MEM_RELEASE);
        }
    }
}

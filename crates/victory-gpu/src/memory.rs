//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

/// What a numbered copy is.
enum Copy {
    /// A title buffer's copy or a window's (or one of them in place).
    Shadow(Arc<Shadow>),
    /// A mirror (MOONSHOT.md, phase 4): a buffer over the whole heap of a
    /// committed section of the page pool, in place, which every view of
    /// the section reaches at its offset in it. Its directory entries are
    /// written for every page the section's views show, as they are mapped,
    /// and cleared when they go: a pointer into such memory reaches it
    /// whether or not anything named it before.
    ///
    /// shortcut: a page a copy holds ([`crate::shadow::copied`]) is left
    /// to the copy, so that a raw address and a root view reach the same
    /// bytes; a copy made after the page was given the mirror is not seen.
    Mirror { section: usize, buffer: ID3D12Resource, bytes: u64 },
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Copy>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        // (mirrors exist only with title memory in place)
        if crate::inplace::in_place_wanted() {
            crate::inplace::follow_views();
        }
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// The directory entry of `page`.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES));
        // SAFETY: inside the directory's mapping.
        unsafe { self.directory.1.add(page as usize * 2) }
    }

    /// Sets an empty entry of `page` to copy `n` (from 0) at byte `offset`.
    fn fill(&self, page: u64, n: usize, offset: u64) {
        let entry = self.entry(page);
        // SAFETY: inside the directory's mapping; the offset first: a
        // shader that sees the number sees it too.
        unsafe {
            if entry.read() == 0 {
                entry.add(1).write_volatile(offset as u32);
                entry.write_volatile(n as u32 + 1);
            }
        }
    }

    /// Gives every page view `[at, at + len)` of mirror `n` shows, from
    /// `offset` in its section, the mirror (pages a copy holds excepted).
    fn fill_view(&self, n: usize, at: usize, len: usize, offset: u64) {
        for (page, offset) in view_pages(at, len, offset) {
            if !crate::shadow::copied(page * PAGE) {
                self.fill(page, n, offset);
            }
        }
    }

    /// The mirror of the section whose view shows `start` (a page), made
    /// now if there is none; with every page of the section's views given
    /// it. False if `start` is no view of a committed section, or a copy
    /// holds it.
    fn mirror(&mut self, start: u64) -> bool {
        let crate::inplace::Cover::Run(section, _) = crate::inplace::cover(start as usize, PAGE as usize) else {
            return false;
        };
        if crate::shadow::copied(start) {
            return false;
        }
        let n = match self.copies.iter().position(|c| matches!(c, Copy::Mirror { section: s, .. } if *s == section)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                let Some((buffer, bytes)) = crate::inplace::mirror(section) else {
                    return false;
                };
                tracing::info!(
                    section = format_args!("{section:#x}"),
                    mb = bytes >> 20,
                    "title memory by address: physical pages reached in place, the whole section"
                );
                self.copies.push(Copy::Mirror { section, buffer, bytes });
                self.copies.len() - 1
            }
            None => return false,
        };
        for (at, len, offset) in crate::inplace::views_of(section) {
            self.fill_view(n, at, len, offset);
        }
        true
    }

    /// The views mapped and gone since the last look: a mirror's entries
    /// where a view went are cleared (a mirror is by section, so they no
    /// longer say what the address shows; the next touch is a fault, and
    /// the page is given what shows there then; a copy is by address and
    /// keeps its entries), a view of a mirrored section is given its
    /// mirror.
    fn follow_views(&mut self) {
        for (at, len, section, offset) in crate::inplace::take_view_changes() {
            if section == 0 {
                for (page, _) in view_pages(at, len, 0) {
                    let entry = self.entry(page);
                    // SAFETY: inside the directory's mapping.
                    let n = unsafe { entry.read() } as usize;
                    if n != 0 && matches!(self.copies.get(n - 1), Some(Copy::Mirror { .. })) {
                        // SAFETY: as above; the number first: a shader that
                        // sees no number reads no offset.
                        unsafe { entry.write_volatile(0) };
                    }
                }
            } else if let Some(n) = self.copies.iter().position(|c| matches!(c, Copy::Mirror { section: s, .. } if *s == section)) {
                self.fill_view(n, at, len, offset);
            }
        }
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // View memory of a committed section: the section's mirror.
        if self.mirror(start) {
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| matches!(c, Copy::Shadow(s) if Arc::ptr_eq(s, &shadow))) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(Copy::Shadow(shadow.clone()));
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below. The
                    // byte in the copy's buffer: in place, the buffer
                    // begins on the page line before the title memory.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile(shadow.offset(at) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset(start);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

/// The directory pages a view `[at, at + len)` showing its section from
/// `offset` on covers, each with the byte of the section it begins at
/// (views lie on 64 KiB lines); none past the directory's last entry,
/// which stays empty.
fn view_pages(at: usize, len: usize, offset: u64) -> impl Iterator<Item = (u64, u64)> {
    let last = u64::from(MEMORY_PAGES) - 1;
    let (from, to) = ((at as u64 / PAGE).min(last), ((at + len) as u64).div_ceil(PAGE).min(last));
    (from..to).map(move |page| (page, offset + page * PAGE - at as u64))
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    m.follow_views();
    let numbered = m.copies.len();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, copy) in m.copies.iter().enumerate().skip(made as usize) {
        match copy {
            // From the buffer's first byte, which in place lies before the
            // title memory's (the directory's offsets count from it).
            Copy::Shadow(shadow) => {
                let (start, len) = shadow.span();
                heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, shadow.offset(start) + len);
            }
            Copy::Mirror { buffer, bytes, .. } => heap.raw_uav_at(base + 1 + n as u32, buffer, 0, *bytes),
        }
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    for (n, copy) in m.copies.iter().enumerate() {
        // A mirror is the title's memory: nothing to bring up to date.
        let Copy::Shadow(shadow) = copy else { continue };
        let (start, len) = shadow.span();
        if first_read || n >= numbered {
            shadow.refresh(list, start, len);
        }
        if writes && (first_write || n >= numbered) {
            shadow.gpu_writes(list, start, len);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view's pages are the section's bytes at the view's offset on, and
    /// the directory's last entry is never given (addresses past the
    /// directory read it).
    #[test]
    fn a_view_gives_each_page_its_byte_in_the_section() {
        let at = 0x40_0000_0000usize; // 256 GiB, past the directory's 128
        let got: Vec<(u64, u64)> = view_pages(0x1_0002_0000, 3 * PAGE as usize, 0x50_0000).collect();
        assert_eq!(got, [(0x1_0002, 0x50_0000), (0x1_0003, 0x51_0000), (0x1_0004, 0x52_0000)]);
        let last = u64::from(MEMORY_PAGES) - 1;
        let near = (last as usize - 1) * PAGE as usize;
        assert_eq!(view_pages(near, 4 * PAGE as usize, 0).map(|(p, _)| p).collect::<Vec<_>>(), [last - 1]);
        assert_eq!(view_pages(at, PAGE as usize, 0).count(), 0, "past the directory");
    }
}

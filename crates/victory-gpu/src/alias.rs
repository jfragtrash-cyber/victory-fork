//! Copies the runtime keeps of physical pages at the targets of the GPU's
//! page-mapping copies (xmem's `VICTORY_ALIAS_COPY=1`): a target page is
//! memory of ours, write-watched, that holds the source page's bytes
//! instead of showing them through a view. Two things keep the two the
//! same, as one set of bytes would be:
//!
//! - [`pull`]: before anything asks what changed in title memory that holds
//!   copies (`crate::watch::writes`), each copy whose source page was
//!   written since the copy was taken is taken again. The copy's own write
//!   watch then says it changed, so whoever asked reads it anew.
//! - [`push`]: what shaders wrote into a copy (put back into title memory,
//!   `crate::host::write_mapped`) is written into its source page too.
//!
//! What DirectStorage writes into physical pages the runtime carries to the
//! copies itself (`xmem::pages_written`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const PAGE: usize = 64 << 10;

/// The runtime's answer: an address where the title shows the page the copy
/// at `page` holds (a view of the title's), if one does.
static SOURCE: OnceLock<fn(usize) -> Option<usize>> = OnceLock::new();

/// A copy: where its source page shows (an address in a view of the
/// title's) and the runtime's change number when that was known; the asking
/// of the write watch and guards (`crate::watch::writes`) after which the
/// source was last taken (0: not known yet, taken again at the first look).
#[derive(Clone, Copy)]
struct Copy {
    source: usize,
    seq: u64,
    asked: u64,
}

/// The copies by address.
static COPIES: Mutex<BTreeMap<usize, Copy>> = Mutex::new(BTreeMap::new());

/// Whether there is any copy (a look at [`COPIES`] costs a lock).
static ANY: AtomicBool = AtomicBool::new(false);

/// The copies by their source's 64 KiB page, for [`source_opened`].
static BY_SOURCE: Mutex<Option<std::collections::HashMap<usize, Vec<usize>>>> = Mutex::new(None);

/// Copies whose source was written (its guard opened) or that are new:
/// taken again, and their texture slots told, at the next submission
/// ([`flush`]).
static DIRTY: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());
static ANY_DIRTY: AtomicBool = AtomicBool::new(false);

/// Notes `page` as the copy of the source page holding `source` (`Some`),
/// or as no copy any more of `old`.
fn by_source(page: usize, old: Option<usize>, new: Option<usize>) {
    let mut by = BY_SOURCE.lock().unwrap_or_else(|e| e.into_inner());
    let by = by.get_or_insert_with(Default::default);
    if let Some(old) = old {
        let key = old - old % PAGE;
        if let Some(list) = by.get_mut(&key) {
            list.retain(|&p| p != page);
            if list.is_empty() {
                by.remove(&key);
            }
        }
    }
    if let Some(new) = new {
        by.entry(new - new % PAGE).or_default().push(page);
    }
}

/// The guard of title memory `[block, block + BLOCK)` opened (the title
/// wrote it, or a write of ours through the kernel comes): copies of pages
/// there are taken again at the next submission. Called with the guards'
/// lock held; takes the copies' locks alone.
pub(crate) fn source_opened(block: usize, len: usize) {
    if !ANY.load(Ordering::Relaxed) {
        return;
    }
    let by = BY_SOURCE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(by) = by.as_ref() else {
        return;
    };
    let mut dirty = None;
    let mut at = block - block % PAGE;
    while at < block + len {
        if let Some(pages) = by.get(&at) {
            dirty.get_or_insert_with(|| DIRTY.lock().unwrap_or_else(|e| e.into_inner())).extend(pages.iter().copied());
            ANY_DIRTY.store(true, Ordering::Relaxed);
        }
        at += PAGE;
    }
}

/// At a submission's start: each copy whose source was written since it was
/// taken, or that is new, is taken again, its source's guard set again (so
/// the next write faults and says so), and the texture slots over it told
/// (`crate::descriptor::host::page_written`), as a write through a view of
/// the page would have.
pub(crate) fn flush() {
    if !ANY_DIRTY.swap(false, Ordering::Relaxed) {
        return;
    }
    let pages = std::mem::take(&mut *DIRTY.lock().unwrap_or_else(|e| e.into_inner()));
    for page in pages {
        let Some(copy) = COPIES.lock().unwrap_or_else(|e| e.into_inner()).get(&page).copied() else {
            continue;
        };
        let seq = crate::watch::remap_seq();
        let Some(from) = source_now(page, &copy) else {
            continue;
        };
        // Sets the source's guard again (and notes the asking).
        let Some((now, _)) = crate::watch::writes(from, from + PAGE, true) else {
            continue;
        };
        let Some(_target) = crate::watch::title_memory_reading_now(page, PAGE) else {
            DIRTY.lock().unwrap_or_else(|e| e.into_inner()).insert(page);
            ANY_DIRTY.store(true, Ordering::Relaxed);
            continue;
        };
        let Some(_source) = crate::watch::title_memory_reading_now(from, PAGE) else {
            DIRTY.lock().unwrap_or_else(|e| e.into_inner()).insert(page);
            ANY_DIRTY.store(true, Ordering::Relaxed);
            continue;
        };
        if crate::watch::remapped_since(seq, page, PAGE) || crate::watch::remapped_since(seq, from, PAGE) {
            continue;
        }
        crate::watch::writing(page, PAGE);
        // SAFETY: both pages are mapped and stay so while the reads are held.
        unsafe { std::ptr::copy_nonoverlapping(from as *const u8, page as *mut u8, PAGE) };
        STATS[1].fetch_add(1, Ordering::Relaxed);
        if let Some(c) = COPIES.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&page) {
            c.asked = now;
        }
        crate::descriptor::host::page_written(page, PAGE);
    }
}

/// Since the last frame line: copies looked at by [`pull`], taken again,
/// and pieces written back to their sources by [`push`].
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The runtime says how to find a copy's source page (once).
pub fn title_memory_copies(source: fn(usize) -> Option<usize>) {
    let _ = SOURCE.set(source);
}

/// The runtime made the 64 KiB page at `page` a copy of the page shown at
/// `source` (`Some`), or it no longer is a copy (`None`).
pub fn title_memory_copied(page: usize, source: Option<usize>) {
    let seq = crate::watch::remap_seq();
    let old = {
        let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let old = match source {
            Some(source) => copies.insert(page, Copy { source, seq, asked: 0 }),
            None => copies.remove(&page),
        };
        ANY.store(!copies.is_empty(), Ordering::Relaxed);
        old
    };
    by_source(page, old.map(|c| c.source), source);
    // A new copy: its source's guard is set at the next submission.
    if source.is_some() {
        DIRTY.lock().unwrap_or_else(|e| e.into_inner()).insert(page);
        ANY_DIRTY.store(true, Ordering::Relaxed);
    }
}

/// Whether the 64 KiB page holding `at` is a copy: the guards take it for
/// view memory (`crate::watch::is_view`). The title's processor never
/// writes such a page (it is the card's alone on the console), and every
/// write of ours into it opens its guard first, so a guard there says
/// exactly when a copy was taken again.
pub(crate) fn is_copy(at: usize) -> bool {
    ANY.load(Ordering::Relaxed) && COPIES.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&(at - at % PAGE))
}

/// The copies that hold bytes of `[start, end)`.
fn copies_in(start: usize, end: usize) -> Vec<(usize, Copy)> {
    if !ANY.load(Ordering::Relaxed) || start >= end {
        return Vec::new();
    }
    let copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
    copies.range(start - start % PAGE..end).map(|(&p, &c)| (p, c)).collect()
}

/// Where the copy at `page` finds its source now: the address it was told,
/// unless the runtime changed what shows there since; then the runtime is
/// asked (and the answer kept).
fn source_now(page: usize, copy: &Copy) -> Option<usize> {
    if !crate::watch::remapped_since(copy.seq, copy.source, PAGE) {
        return Some(copy.source);
    }
    let seq = crate::watch::remap_seq();
    let source = (SOURCE.get()?)(page)?;
    let old = {
        let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        copies.get_mut(&page).map(|c| std::mem::replace(&mut c.source, source)).inspect(|_| {
            if let Some(c) = copies.get_mut(&page) {
                c.seq = seq;
            }
        })
    };
    if old != Some(source) {
        by_source(page, old, Some(source));
    }
    Some(source)
}

/// Before `[start, end)` is asked about: the copies there whose source page
/// was written since they were taken are taken again. A copy whose memory
/// or source the runtime is remapping right now is left for the next look
/// (no wait: the caller may hold reads of its own).
pub(crate) fn pull(start: usize, end: usize) {
    let copies = copies_in(start, end);
    if copies.is_empty() {
        return;
    }
    for (page, copy) in copies {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        // The runtime's change number first: a change of the copy or of its
        // source after it is seen below (the runtime's lock is not taken
        // while reads are held: it waits for reads under that lock).
        let seq = crate::watch::remap_seq();
        let Some(from) = source_now(page, &copy) else {
            continue;
        };
        let asked = copy.asked;
        // The source's guard still stands untouched since it was set: not
        // written since the copy was last looked at (a look at a set).
        if asked != 0 && crate::watch::block_clean(from) {
            continue;
        }
        let Some((now, written)) = crate::watch::writes(from, from + PAGE, true) else {
            continue;
        };
        if asked != 0 && written.iter().all(|&(_, w)| w <= asked) {
            let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(c) = copies.get_mut(&page) {
                c.asked = now;
            }
            continue;
        }
        let Some(_target) = crate::watch::title_memory_reading_now(page, PAGE) else {
            continue;
        };
        let Some(_source) = crate::watch::title_memory_reading_now(from, PAGE) else {
            continue;
        };
        // Still the same copy and source: the runtime changes either only
        // with the remapping lock, which the two reads keep out from here.
        if crate::watch::remapped_since(seq, page, PAGE) || crate::watch::remapped_since(seq, from, PAGE) {
            continue;
        }
        // Its guard opens first (`is_copy`).
        crate::watch::writing(page, PAGE);
        // SAFETY: both pages are mapped (a view of the title's, a page of
        // ours) and stay so while the reads are held.
        unsafe { std::ptr::copy_nonoverlapping(from as *const u8, page as *mut u8, PAGE) };
        STATS[1].fetch_add(1, Ordering::Relaxed);
        let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = copies.get_mut(&page) {
            c.asked = now;
        }
    }
}

/// After shaders' bytes were put into title memory `[at, at + len)`: the
/// pieces of it in copies are written into their source pages too.
pub(crate) fn push(at: usize, len: usize) {
    let copies = copies_in(at, at.saturating_add(len));
    if copies.is_empty() {
        return;
    }
    let end = at + len;
    for (page, copy) in copies {
        let seq = crate::watch::remap_seq();
        let Some(from) = source_now(page, &copy) else {
            continue;
        };
        let (lo, hi) = (at.max(page), end.min(page + PAGE));
        if lo >= hi {
            continue;
        }
        let Some(_target) = crate::watch::title_memory_reading_now(page, PAGE) else {
            continue;
        };
        let Some(_source) = crate::watch::title_memory_reading_now(from, PAGE) else {
            continue;
        };
        if crate::watch::remapped_since(seq, page, PAGE) || crate::watch::remapped_since(seq, from, PAGE) {
            continue;
        }
        let to = from + (lo - page);
        // The source's guards open first, as for any write of ours.
        crate::watch::writing(to, hi - lo);
        crate::host::write_mapped_plain(to as *mut u8, lo as *const u8, hi - lo);
        STATS[2].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};

    /// The test's copies' sources (the runtime keeps its own).
    static SOURCES: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

    fn source_of(page: usize) -> Option<usize> {
        SOURCES.lock().unwrap().get(&page).copied()
    }

    /// A copy follows its source when looked at, and shaders' bytes put
    /// into it reach the source.
    #[test]
    fn copies_follow_their_sources() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: fresh write-watched pages of ours, freed at the end.
        let alloc = || unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        let (source, copy) = (alloc(), alloc());
        assert!(source != 0 && copy != 0);
        title_memory_copies(source_of);
        SOURCES.lock().unwrap().insert(copy, source);
        title_memory_copied(copy, Some(source));
        // SAFETY: (all below) our committed pages.
        unsafe {
            (source as *mut u8).add(10).write(5);
            assert!(crate::watch::writes(copy, copy + PAGE, false).is_some());
            assert_eq!((copy as *const u8).add(10).read(), 5, "taken at the first look");
            (source as *mut u8).add(11).write(6);
            crate::watch::writes(copy, copy + PAGE, false);
            assert_eq!((copy as *const u8).add(11).read(), 6, "taken again after its source was written");
            // Shaders' bytes into the copy.
            let bytes = [9u8; 4];
            write_mapped_test(copy + 100, &bytes);
            assert_eq!((source as *const u8).add(101).read(), 9, "reached the source");
        }
        title_memory_copied(copy, None);
        SOURCES.lock().unwrap().remove(&copy);
        // SAFETY: ours.
        unsafe {
            VirtualFree(source as *mut _, 0, MEM_RELEASE);
            VirtualFree(copy as *mut _, 0, MEM_RELEASE);
        }
    }

    fn write_mapped_test(at: usize, bytes: &[u8]) {
        crate::host::write_mapped(at as *mut u8, bytes.as_ptr(), bytes.len());
    }
}

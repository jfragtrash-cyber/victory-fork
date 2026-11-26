//! The title's page-mapping copies (`CopyPageMappingsBatchX`, how it
//! streams textures) as queue work: applied by a thread of ours when the
//! queue's line gets to them, while the title's thread goes on
//! (`VICTORY_ASYNC_PAGE_MAPPINGS=1`; off by default, needs
//! `VICTORY_ASYNC_SUBMIT=1`).
//!
//! Why: the batches were applied on the title's thread, inside its call,
//! one page a call to the runtime's mapper, each an unmap and a map: 120 to
//! 1,000 us a page in the running title, about 30 us in a quiet process
//! (xmem's `page_mapping_cost`). Cutscene frames of 1 to 3.5 s carried
//! 2,000 to 4,800 of them, 0.3 to 2.9 s of the title's thread (run R's
//! frame log). On the console a batch is a queue operation: the card
//! applies it in order with the queue's other work, and the call returns
//! at once.
//!
//! The order kept:
//! - A batch is one more thing in its queue's line (`crate::queue`'s
//!   `Held::Map`). The line hands it to the thread here ([`worker`]) when
//!   it gets there, that is once every submission before it has been
//!   prepared (preparing reads title memory at the addresses the batch
//!   changes) and sent; nothing behind it in its line is looked at before
//!   it is applied, so the submissions behind it are prepared with the new
//!   mappings and the fences signalled behind it are set after it. A line
//!   that holds nothing hands it at once.
//! - Other queues' lines go on: on the console only fences order work
//!   across queues, and a fence signalled behind the batch is set after
//!   it.
//! - Flights in the air whose shaders may write the batch's targets land
//!   first (`crate::host::land_writes_in`): what they wrote goes to the
//!   pages they were prepared against. (Applied in the title's call it
//!   went wherever the targets showed by then.)
//! - A present that forces the lines (`VICTORY_WAIT_FORCE`, on) applies a
//!   batch it meets on its own thread, in its place, when a submission or
//!   a present is behind it in its line, which may be part of the frame
//!   shown ([`force`]); a batch with only signals, waits or batches behind
//!   it is left to our thread, and its line goes on when it is done (the
//!   render thread would otherwise wait for a streaming queue's remapping).
//!
//! Not ahead of earlier submissions whose reads miss its ranges: what a
//! submission reads is not known before it is prepared. Its textures are
//! found through descriptor heaps in title memory, read while preparing;
//! a host list's recorded steps name buffer copies and fills only.
//!
//! The title's processor: it uses the targets too (run AA, with copies at
//! the targets, ended on a breakpoint of the title's). Until a batch is
//! applied the title reads and writes what its targets show now. That is
//! the console's order as well: the call returns before the card gets to
//! the batch, and the title orders its processor after the card with
//! fences, which are set after the batch here. Where a target shows
//! nothing yet the access would fault, and a race the console wins in
//! microseconds would crash here: such an access, and one that meets a
//! page in the moment it is remapped, waits until the batches over that
//! page are applied ([`fault_at`]). A mapping the title makes itself
//! (`XMemMapPhysicalPages`) over a batch's targets or sources applies the
//! batches under it first, on the title's thread
//! ([`title_memory_mapping`]): mappings change in the order they were asked
//! for, as when every batch was applied in its call. Either applies a
//! batch ahead of its place in its line when the line had not got there:
//! counted.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// One call of the page mapper (`crate::resource::PageMapper`): `(to,
/// from, len)`, `from` 0 unmapping.
pub(crate) type Op = (u64, u64, u64);

const PAGE: u64 = 64 << 10;

/// In its queue's line, not got to yet.
const PENDING: u8 = 0;
/// Given to [`worker`].
const HANDED: u8 = 1;
/// Being applied (by [`worker`], or by a title thread that may not wait).
const APPLYING: u8 = 2;
const DONE: u8 = 3;

/// One `CopyPageMappingsBatchX` call of the title's.
pub(crate) struct Batch {
    ops: Vec<Op>,
    /// The title memory it changes, and the memory it maps from: `[from,
    /// to)`, in order and joined.
    targets: Vec<(usize, usize)>,
    sources: Vec<(usize, usize)>,
    pages: u64,
    state: AtomicU8,
    called: Instant,
}

impl Batch {
    pub(crate) fn done(&self) -> bool {
        self.state.load(Ordering::Acquire) == DONE
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether batches go behind their queue's line (`VICTORY_ASYNC_PAGE_MAPPINGS=1`;
/// needs `VICTORY_ASYNC_SUBMIT=1`, whose thread runs the lines). Off by
/// default: on once a cutscene run shows the title's thread freed of the
/// remapping time with the picture right (the frame log's "page
/// mappings").
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_ASYNC_PAGE_MAPPINGS").is_some_and(|v| v == "1");
        let on = asked && crate::queue::behind() && crate::resource::page_mapper().is_some();
        if asked && !on {
            tracing::warn!("VICTORY_ASYNC_PAGE_MAPPINGS=1 needs VICTORY_ASYNC_SUBMIT=1 (and its defaults): page mappings stay in the title's calls");
        }
        if on {
            start();
            tracing::info!(
                "the title's page-mapping copies are applied by a thread of ours in order with their queue's work, not in its calls (VICTORY_ASYNC_PAGE_MAPPINGS=1)"
            );
        }
        on
    })
}

/// Starts [`worker`] and puts the fault handler in place, once.
fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-remap".into()).spawn(worker) {
            crate::host::fail(format_args!("no thread for page mappings (VICTORY_ASYNC_PAGE_MAPPINGS=1): {e}"));
        }
        install();
    });
}

/// The batches not applied yet, in call order.
static UNAPPLIED: Mutex<Vec<Arc<Batch>>> = Mutex::new(Vec::new());
/// How many there are (read without the lock: the fault handler's first
/// look, and the title's own mappings').
static HOW_MANY: AtomicUsize = AtomicUsize::new(0);

/// What [`worker`] is given: batches whose line got to them, in that
/// order, and batches an access of the title's waits for, first.
struct Work {
    line: VecDeque<Arc<Batch>>,
    urgent: VecDeque<Arc<Batch>>,
}

static WORK: Mutex<Work> = Mutex::new(Work { line: VecDeque::new(), urgent: VecDeque::new() });
static WORK_MOVED: Condvar = Condvar::new();
/// Told whenever a batch is done.
static FINISHED: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

std::thread_local! {
    /// This thread is applying a batch: its own faults are not waited for.
    static APPLYING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `ranges` in order, joined where they overlap or touch.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

fn meets(ranges: &[(usize, usize)], (from, to): (usize, usize)) -> bool {
    ranges.iter().any(|&(a, b)| a < to && from < b)
}

/// A batch of the title's, noted as not applied yet (behind the batches
/// asked for before it).
pub(crate) fn batch(ops: Vec<Op>) -> Arc<Batch> {
    let span = |at: u64, len: u64| (at as usize, at.saturating_add(len) as usize);
    let targets = joined(ops.iter().map(|&(to, _, len)| span(to, len)).collect());
    let sources = joined(ops.iter().filter(|op| op.1 != 0).map(|&(_, from, len)| span(from, len)).collect());
    let pages = ops.iter().map(|op| op.2.div_ceil(PAGE)).sum();
    let batch = Arc::new(Batch { ops, targets, sources, pages, state: AtomicU8::new(PENDING), called: Instant::now() });
    let mut unapplied = lock(&UNAPPLIED);
    unapplied.push(batch.clone());
    HOW_MANY.store(unapplied.len(), Ordering::Release);
    batch
}

/// The line got to `batch`: [`worker`] applies it, behind the batches
/// handed before it.
pub(crate) fn hand(batch: &Arc<Batch>) {
    if batch.state.compare_exchange(PENDING, HANDED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        lock(&WORK).line.push_back(batch.clone());
        WORK_MOVED.notify_one();
    }
}

/// Takes `batch` to apply it: the state it had, if it was not taken yet.
fn claim(batch: &Batch) -> Option<u8> {
    let mut state = batch.state.load(Ordering::Acquire);
    while state < APPLYING {
        match batch.state.compare_exchange(state, APPLYING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Some(state),
            Err(now) => state = now,
        }
    }
    None
}

/// Who applied a batch, for the counts.
#[derive(Clone, Copy)]
enum By {
    /// [`worker`], its line having got to it.
    Line,
    /// [`worker`], for an access of the title's that faulted.
    Fault,
    /// A title thread, before a mapping of its own.
    Mapping,
    /// The thread of a present that forced the lines.
    Forced,
}

/// Since the last queue line ([`STATS`]' places).
const S_LINE: usize = 0;
const S_FAULT: usize = 1;
const S_MAPPING: usize = 2;
const S_FORCED: usize = 3;
const S_PAGES: usize = 4;
const S_APPLY_US: usize = 5;
const S_LAG_US: usize = 6;
const S_EARLY: usize = 7;
const S_WAIT_US: usize = 8;
const S_CALLS: usize = 9;
const S_CALL_US: usize = 10;

/// Since the last queue line: batches applied by [`worker`] as their line
/// got to them and for an access that faulted; batches applied on a title
/// thread before a mapping of its own and at a present that forced the
/// lines; their pages; the microseconds applying them; the microseconds
/// from their call to their end; batches applied before their line got to
/// them; microseconds title threads waited for batches (forced, faulted,
/// before their own mappings); the title's calls and its microseconds in
/// them.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// Applies `batch`, taken with [`claim`] (`was`: its state before).
fn apply(batch: &Arc<Batch>, by: By, was: u8) {
    let started = Instant::now();
    let before = APPLYING_HERE.replace(true);
    // What flights in the air write there goes to the pages shown now.
    crate::host::land_writes_in(&batch.targets);
    if let Some(map) = crate::resource::page_mapper() {
        let mut ok = true;
        for &(to, from, len) in &batch.ops {
            ok &= map(to, from, len);
        }
        if !ok {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a page mapping failed (applied by a thread of ours: VICTORY_ASYNC_PAGE_MAPPINGS=1)"));
        }
    }
    APPLYING_HERE.set(before);
    finish(batch, by, was, started);
}

fn finish(batch: &Arc<Batch>, by: By, was: u8, started: Instant) {
    batch.state.store(DONE, Ordering::Release);
    {
        let mut unapplied = lock(&UNAPPLIED);
        if let Some(i) = unapplied.iter().position(|b| Arc::ptr_eq(b, batch)) {
            unapplied.remove(i);
        }
        HOW_MANY.store(unapplied.len(), Ordering::Release);
    }
    drop(lock(&FINISHED.0));
    FINISHED.1.notify_all();
    let (us, lag) = (started.elapsed().as_micros() as u64, batch.called.elapsed().as_micros() as u64);
    let early = was == PENDING && matches!(by, By::Fault | By::Mapping);
    STATS[match by {
        By::Line => S_LINE,
        By::Fault => S_FAULT,
        By::Mapping => S_MAPPING,
        By::Forced => S_FORCED,
    }]
    .fetch_add(1, Ordering::Relaxed);
    STATS[S_PAGES].fetch_add(batch.pages, Ordering::Relaxed);
    STATS[S_APPLY_US].fetch_add(us, Ordering::Relaxed);
    STATS[S_LAG_US].fetch_add(lag, Ordering::Relaxed);
    STATS[S_EARLY].fetch_add(u64::from(early), Ordering::Relaxed);
    if matches!(by, By::Line | By::Fault) {
        crate::framelog::add(crate::framelog::T::PageMapBatches, 1);
        crate::framelog::add(crate::framelog::T::PageMapPages, batch.pages);
        crate::framelog::add(crate::framelog::T::PageMapApplyUs, us);
        crate::framelog::add(crate::framelog::T::PageMapLagUs, lag);
    }
    if early {
        crate::framelog::add(crate::framelog::T::PageMapEarly, 1);
    }
    line_now();
    // The line behind it may go on.
    crate::queue::mapped();
}

/// Every 1,024 batches applied: what [`STATS`] counted since the last
/// time.
fn line_now() {
    let applied: u64 = STATS[S_LINE..=S_FORCED].iter().map(|n| n.load(Ordering::Relaxed)).sum();
    if applied < 1024 {
        return;
    }
    let n: Vec<u64> = STATS.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let batches: u64 = n[S_LINE..=S_FORCED].iter().sum();
    if batches == 0 {
        return;
    }
    tracing::info!(
        "page mappings (VICTORY_ASYNC_PAGE_MAPPINGS=1): {batches} batches of {} pages applied, {} by our thread as their line got to them, {} by it for an access of the title's that faulted, {} on a title thread before a mapping of its own, {} at a present that forced the lines; {:.1} ms applying them, {:.1} ms from the call to the end on average, {} ahead of their place in the line; the title waited {:.1} ms for batches and spent {:.1} ms in {} calls",
        n[S_PAGES],
        n[S_LINE],
        n[S_FAULT],
        n[S_MAPPING],
        n[S_FORCED],
        n[S_APPLY_US] as f64 / 1e3,
        n[S_LAG_US] as f64 / 1e3 / batches as f64,
        n[S_EARLY],
        n[S_WAIT_US] as f64 / 1e3,
        n[S_CALL_US] as f64 / 1e3,
        n[S_CALLS]
    );
}

/// The title's `CopyPageMappingsBatchX` call took `us` microseconds of its
/// thread.
pub(crate) fn called(us: u64) {
    STATS[S_CALLS].fetch_add(1, Ordering::Relaxed);
    STATS[S_CALL_US].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::PageMapCalls, 1);
    crate::framelog::add(crate::framelog::T::PageMapCallUs, us);
}

/// A title thread waited `since` for batches.
fn waited(since: Instant) {
    let us = since.elapsed().as_micros() as u64;
    STATS[S_WAIT_US].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::PageMapWaitUs, us);
}

/// Waits until `batch` is done, `most` at the longest: whether it is.
fn wait_done(batch: &Batch, most: Duration) -> bool {
    if batch.done() {
        return true;
    }
    let until = Instant::now() + most;
    let mut finished = lock(&FINISHED.0);
    while !batch.done() {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        finished = FINISHED.1.wait_timeout(finished, left).unwrap_or_else(|e| e.into_inner()).0;
    }
    true
}

/// How long a title thread waits for a batch before it gives up and says
/// so: [`worker`] needs nothing of the title's threads, so only a lock of
/// ours held by the waiting thread would keep it.
const WAIT_MOST: Duration = Duration::from_secs(10);

/// Applies `batch` on this thread, or waits until whoever took it is done.
fn here(batch: &Arc<Batch>, by: By) {
    match claim(batch) {
        Some(was) => apply(batch, by, was),
        None => {
            if !wait_done(batch, WAIT_MOST) {
                tracing::warn!("a page-mapping batch was waited for {WAIT_MOST:?} and is not done: going on without it");
            }
        }
    }
}

/// A present forces the lines and has got to `batch`: applied now, on its
/// thread.
pub(crate) fn force(batch: &Arc<Batch>) {
    if batch.done() {
        return;
    }
    let waiting = Instant::now();
    here(batch, By::Forced);
    waited(waiting);
}

/// The batches not applied yet that must be before an access to title
/// memory `[at, at + len)`, or (`sources`) before the title changes what
/// it shows: those whose targets meet it (or, `sources`, whose sources
/// do), and before each of those the earlier ones whose targets meet its
/// targets or its sources (they change what it maps over, or what it maps
/// from, before it). In call order.
fn needed(at: usize, len: usize, sources: bool) -> Vec<Arc<Batch>> {
    let first = (at, at.saturating_add(len.max(1)));
    let unapplied = lock(&UNAPPLIED);
    let mut wanted = vec![first];
    let mut out = Vec::new();
    for b in unapplied.iter().rev() {
        if b.targets.iter().any(|&t| meets(&wanted, t)) || (sources && meets(&b.sources, first)) {
            wanted.extend(b.targets.iter().chain(&b.sources).copied());
            out.push(b.clone());
        }
    }
    out.reverse();
    out
}

/// Before the title maps or unmaps physical pages itself at title memory
/// `[at, at + len)` (`xmem`'s `XMemMapPhysicalPages`, without its locks):
/// the batches not applied yet over that memory, or mapping from it, are
/// applied first, on this thread, so that mappings change in the order
/// they were asked for. Nothing to do (one atomic look) while no batch
/// waits.
pub fn title_memory_mapping(at: usize, len: usize) {
    if HOW_MANY.load(Ordering::Acquire) == 0 {
        return;
    }
    let needed = needed(at, len, true);
    if needed.is_empty() {
        return;
    }
    let waiting = Instant::now();
    for b in &needed {
        here(b, By::Mapping);
    }
    waited(waiting);
}

/// The thread that applies the batches handed to it: those an access waits
/// for first, then those of the lines in the order the lines got to them.
fn worker() {
    crate::framelog::keep_on_fast_cores("victory-remap");
    loop {
        let (batch, by) = {
            let mut work = lock(&WORK);
            loop {
                if let Some(b) = work.urgent.pop_front() {
                    break (b, By::Fault);
                }
                if let Some(b) = work.line.pop_front() {
                    break (b, By::Line);
                }
                work = WORK_MOVED.wait(work).unwrap_or_else(|e| e.into_inner());
            }
        };
        if let Some(was) = claim(&batch) {
            apply(&batch, by, was);
        }
    }
}

/// Whether `at` is committed memory (a guard or another protection
/// faulted there: not ours to wait for).
fn committed(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT
}

/// An access to title memory faulted at `at` where nothing is committed
/// (`EXCEPTION_ACCESS_VIOLATION`): if batches not applied yet are over
/// that page, [`worker`] applies them first and this waits for it; true
/// if the access may be tried again.
fn fault_at(at: usize) -> bool {
    if HOW_MANY.load(Ordering::Acquire) == 0 || APPLYING_HERE.get() {
        return false;
    }
    let page = at - at % PAGE as usize;
    // A first look that allocates nothing: most faults here are the
    // guards' and meet no batch.
    if !lock(&UNAPPLIED).iter().any(|b| meets(&b.targets, (page, page + PAGE as usize))) || committed(at) {
        return false;
    }
    let needed = needed(page, PAGE as usize, false);
    if needed.is_empty() {
        return false;
    }
    let waiting = Instant::now();
    {
        let mut work = lock(&WORK);
        work.urgent.extend(needed.iter().filter(|b| b.state.load(Ordering::Acquire) < APPLYING).cloned());
    }
    WORK_MOVED.notify_one();
    let done = needed.iter().all(|b| wait_done(b, WAIT_MOST));
    waited(waiting);
    if !done {
        tracing::warn!(
            address = format_args!("{at:#x}"),
            "an access to a page-mapping target waited {WAIT_MOST:?} for its batch, which is not done: the fault goes on as it is"
        );
    }
    done
}

/// Registers the handler that takes access violations to [`fault_at`],
/// ahead of the guards' handler and the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    /// [`fault_at`] for access violations, read or write.
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        if HOW_MANY.load(Ordering::Relaxed) == 0 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if fault_at(r.ExceptionInformation[1]) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
    }
    // SAFETY: the handler only looks at our batches and waits for them.
    unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The page mapper of the tests: notes each call, with the value of
    /// the fence watched for its target then ([`watch`]), and commits and
    /// marks memory at targets [`real`] names.
    pub(crate) fn mapper(to: u64, from: u64, len: u64) -> bool {
        let fence = lock(&FENCES).iter().find(|f| f.0 == to).map(|f| f.1);
        // SAFETY: a fence of the test's, alive for the whole test binary.
        let value = fence.map(|f| crate::fence::completed(f as crate::object::This));
        lock(&CALLS).push((to, from, len, value));
        if lock(&REAL).contains(&to) {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
            // SAFETY: a reservation of the test's.
            let at = unsafe { VirtualAlloc(to as *const _, len as usize, MEM_COMMIT, PAGE_READWRITE) };
            if at.is_null() {
                return false;
            }
            // SAFETY: committed just now.
            unsafe { at.cast::<u8>().write(0x5a) };
        }
        true
    }

    static CALLS: Mutex<Vec<(u64, u64, u64, Option<u64>)>> = Mutex::new(Vec::new());
    static FENCES: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
    static REAL: Mutex<Vec<u64>> = Mutex::new(Vec::new());

    /// The tests' mapper is the page mapper (once for the test binary).
    pub(crate) fn use_mapper() {
        crate::resource::use_page_mapper(mapper);
        let ours: crate::resource::PageMapper = mapper;
        assert_eq!(crate::resource::page_mapper().map(|m| m as usize), Some(ours as usize), "another page mapper is set");
    }

    /// The mapper notes `fence`'s value when it maps at `to`.
    pub(crate) fn watch(to: u64, fence: crate::object::This) {
        lock(&FENCES).push((to, fence as usize));
    }

    /// The mapper's calls at `to`, with the watched fence's value then.
    pub(crate) fn calls_at(to: u64) -> Vec<Option<u64>> {
        lock(&CALLS).iter().filter(|c| c.0 == to).map(|c| c.3).collect()
    }

    /// Applies `batch` as [`worker`] would (or waits for whoever took it).
    pub(crate) fn run(batch: &Arc<Batch>) {
        here(batch, By::Line);
    }

    pub(crate) fn state(batch: &Batch) -> u8 {
        batch.state.load(Ordering::Acquire)
    }

    pub(crate) const STATE_PENDING: u8 = PENDING;

    /// Test addresses: never dereferenced unless [`REAL`] names them.
    const BASE: u64 = 0x7e00_0000_0000;

    #[test]
    fn a_batch_knows_its_ranges_and_pages() {
        let b = batch(vec![(BASE, BASE + 0x100 * PAGE, 2 * PAGE), (BASE + 2 * PAGE, 0, PAGE), (BASE + 9 * PAGE, BASE + 0x100 * PAGE, PAGE)]);
        let at = |k: u64| (BASE + k * PAGE) as usize;
        assert_eq!(b.targets, [(at(0), at(3)), (at(9), at(10))], "joined, in order");
        assert_eq!(b.sources, [(at(0x100), at(0x102))], "an unmapping has no source");
        assert_eq!(b.pages, 4);
        use_mapper();
        run(&b);
        assert!(b.done());
    }

    /// An access, or a mapping of the title's, needs the batches over its
    /// memory and, before them, those that change what they map from.
    #[test]
    fn what_an_access_needs_is_found_in_call_order() {
        let base = BASE + 0x1000 * PAGE;
        let at = |k: u64| base + k * PAGE;
        // a maps over the pages b maps from; b maps T; c maps elsewhere; d maps T again.
        let a = batch(vec![(at(10), 0, PAGE)]);
        let b = batch(vec![(at(0), at(10), PAGE)]);
        let c = batch(vec![(at(50), at(60), PAGE)]);
        let d = batch(vec![(at(0), at(20), PAGE)]);
        let names = |v: Vec<Arc<Batch>>| -> Vec<&str> {
            v.iter()
                .map(|x| [(&a, "a"), (&b, "b"), (&c, "c"), (&d, "d")].iter().find(|(y, _)| Arc::ptr_eq(x, y)).map_or("?", |p| p.1))
                .collect()
        };
        assert_eq!(names(needed(at(0) as usize + 5, 1, false)), ["a", "b", "d"]);
        assert_eq!(names(needed(at(50) as usize, PAGE as usize, false)), ["c"]);
        assert!(needed(at(60) as usize, PAGE as usize, false).is_empty(), "a source is not an access's");
        assert_eq!(names(needed(at(60) as usize, PAGE as usize, true)), ["c"], "the title maps over c's source");
        assert_eq!(names(needed(at(20) as usize, PAGE as usize, true)), ["a", "b", "d"]);
        use_mapper();
        for x in [&a, &b, &c, &d] {
            run(x);
        }
        assert!(needed(at(0) as usize, PAGE as usize, true).is_empty(), "done: nothing needed");
    }

    /// The title maps memory a batch maps from: the batch is applied first,
    /// on the title's thread, ahead of its line.
    #[test]
    fn the_titles_own_mapping_applies_the_batches_under_it_first() {
        use_mapper();
        let base = BASE + 0x2000 * PAGE;
        let b = batch(vec![(base, base + 0x40 * PAGE, PAGE)]);
        let other = batch(vec![(base + 0x80 * PAGE, 0, PAGE)]);
        title_memory_mapping((base + 0x40 * PAGE) as usize + 100, 10);
        assert!(b.done(), "applied before the title's mapping");
        assert_eq!(calls_at(base), [None]);
        assert!(!other.done(), "a batch elsewhere waits for its line");
        run(&other);
    }

    /// A read of a target that shows nothing yet waits for its batch: the
    /// fault handler has the thread here apply it, and the read is tried
    /// again.
    #[test]
    fn a_read_of_a_target_not_mapped_yet_waits_for_its_batch() {
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        use_mapper();
        start();
        // A fresh reservation of ours, kept: given back, its addresses go to
        // the guards' tests, whose state by address does not like that.
        // SAFETY: a plain reservation.
        let at = unsafe { VirtualAlloc(std::ptr::null(), PAGE as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(at, 0);
        lock(&REAL).push(at);
        let b = batch(vec![(at, 0x1000, PAGE)]);
        assert_eq!(state(&b), PENDING, "its line has not got to it");
        // SAFETY: reserved, not committed: the read faults, and the
        // handler commits it through the batch.
        let read = unsafe { std::ptr::read_volatile(at as *const u8) };
        assert_eq!(read, 0x5a, "the read saw the page the batch maps");
        assert!(b.done());
    }

    /// Forced lines apply a batch on their own thread, its line having got
    /// to it; one done already is left.
    #[test]
    fn a_forced_batch_is_applied_on_the_thread_that_forces() {
        use_mapper();
        let to = BASE + 0x3000 * PAGE;
        let b = batch(vec![(to, 0, PAGE)]);
        force(&b);
        assert!(b.done());
        force(&b);
        assert_eq!(calls_at(to), [None], "applied once");
    }

    #[test]
    fn ranges_join_and_meet() {
        assert_eq!(joined(vec![(5, 9), (0, 2), (2, 4), (8, 12), (20, 20)]), [(0, 4), (5, 12)]);
        assert!(meets(&[(0, 4), (5, 12)], (3, 6)));
        assert!(!meets(&[(0, 4), (5, 12)], (4, 5)));
    }
}

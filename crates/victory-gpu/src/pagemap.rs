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
//! nothing yet the title's access would fault, and a race the console wins
//! in microseconds would crash here: the faulting thread applies the
//! mappings of that page itself, out of their batch's turn, and goes on
//! ([`fault_at`]). A mapping the title makes itself (`XMemMapPhysicalPages`)
//! over a batch's targets or sources applies those mappings first, the
//! same way ([`title_memory_mapping`]): mappings change in the order they
//! were asked for, as when every batch was applied in its call. Either
//! counts the pages it applied before their line got to them.
//!
//! Only the title's own code is helped so, and only on a thread that
//! holds no reading of title memory in place: run AH's submitting thread
//! read a texture in place under the range lock with a copy that expects
//! faults on pages not there (`crate::plaincopy`), faulted on a page of a
//! batch not applied yet, and waited here for the batch, whose remapping
//! waited for that reading; 10 s each, until the wait gave up, and the
//! readers behind the waiting remapping (DirectStorage's workers) stood
//! as long. Our code that faults on title memory has its own answer to
//! it, and holds locks a remapping may need.

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
/// Being applied whole (by [`worker`], or at a present that forced it).
const APPLYING: u8 = 2;
const DONE: u8 = 3;

/// A mapping of a batch: not applied, being applied, applied.
const OP_FREE: u8 = 0;
const OP_APPLYING: u8 = 1;
const OP_DONE: u8 = 2;

/// One `CopyPageMappingsBatchX` call of the title's.
pub(crate) struct Batch {
    ops: Vec<Op>,
    /// Each op's own state: an access of the title's may apply the ops of
    /// its page before the batch's turn ([`fault_at`]).
    op_state: Vec<AtomicU8>,
    /// The title memory it changes, and the memory it maps from: `[from,
    /// to)`, in order and joined.
    targets: Vec<(usize, usize)>,
    sources: Vec<(usize, usize)>,
    pages: u64,
    state: AtomicU8,
    called: Instant,
}

impl Batch {
    /// Applied whole: what is behind it in its line may run.
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
                title_code = TITLE_CODE.get().is_some(),
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

/// The title's code, `[from, to)`: faults there are the title's own
/// accesses ([`fault_at`]).
static TITLE_CODE: OnceLock<(usize, usize)> = OnceLock::new();

/// The runtime says where the title's main image is (`len` bytes from
/// `base`), once: an access to a page-mapping target not applied yet from
/// code there is helped ([`fault_at`]); from anywhere else, never.
pub fn title_code(base: usize, len: usize) {
    let _ = TITLE_CODE.set((base, base.saturating_add(len)));
}

fn in_title_code(at: usize) -> bool {
    TITLE_CODE.get().is_some_and(|&(from, to)| (from..to).contains(&at))
}

/// The batches not applied whole yet, in call order.
static UNAPPLIED: Mutex<Vec<Arc<Batch>>> = Mutex::new(Vec::new());
/// How many there are (read without the lock: the fault handler's first
/// look, and the title's own mappings').
static HOW_MANY: AtomicUsize = AtomicUsize::new(0);

/// What [`worker`] is given: batches whose line got to them, in that
/// order.
static WORK: Mutex<VecDeque<Arc<Batch>>> = Mutex::new(VecDeque::new());
static WORK_MOVED: Condvar = Condvar::new();
/// Told whenever an op or a batch is done.
static FINISHED: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

std::thread_local! {
    /// This thread is applying mappings: its own faults are not ours to
    /// help.
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

fn span(at: u64, len: u64) -> (usize, usize) {
    (at as usize, at.saturating_add(len) as usize)
}

/// What an op changes, and what it maps from (none for an unmapping).
fn op_ranges(&(to, from, len): &Op) -> ((usize, usize), Option<(usize, usize)>) {
    (span(to, len), (from != 0).then(|| span(from, len)))
}

/// A batch of the title's, noted as not applied yet (behind the batches
/// asked for before it).
pub(crate) fn batch(ops: Vec<Op>) -> Arc<Batch> {
    let targets = joined(ops.iter().map(|&(to, _, len)| span(to, len)).collect());
    let sources = joined(ops.iter().filter(|op| op.1 != 0).map(|&(_, from, len)| span(from, len)).collect());
    let pages = ops.iter().map(|op| op.2.div_ceil(PAGE)).sum();
    let op_state = ops.iter().map(|_| AtomicU8::new(OP_FREE)).collect();
    let batch = Arc::new(Batch { ops, op_state, targets, sources, pages, state: AtomicU8::new(PENDING), called: Instant::now() });
    let mut unapplied = lock(&UNAPPLIED);
    unapplied.push(batch.clone());
    HOW_MANY.store(unapplied.len(), Ordering::Release);
    batch
}

/// The line got to `batch`: [`worker`] applies it, behind the batches
/// handed before it.
pub(crate) fn hand(batch: &Arc<Batch>) {
    if batch.state.compare_exchange(PENDING, HANDED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        lock(&WORK).push_back(batch.clone());
        WORK_MOVED.notify_one();
    }
}

/// Takes `batch` to apply it whole: whether it was not taken yet.
fn claim(batch: &Batch) -> bool {
    let mut state = batch.state.load(Ordering::Acquire);
    while state < APPLYING {
        match batch.state.compare_exchange(state, APPLYING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => state = now,
        }
    }
    false
}

/// Applies op `i` of `batch` on this thread, unless another thread has
/// taken it; then waits until that one is done with it. Whether it was
/// applied here.
fn run_op(batch: &Batch, i: usize) -> bool {
    if batch.op_state[i].compare_exchange(OP_FREE, OP_APPLYING, Ordering::AcqRel, Ordering::Acquire).is_err() {
        // Taken by a thread that is applying it now and waits for nothing
        // of this one's ([`fault_at`]'s conditions).
        wait(|| batch.op_state[i].load(Ordering::Acquire) == OP_DONE, "a page mapping being applied by another thread");
        return false;
    }
    let before = APPLYING_HERE.replace(true);
    if let Some(map) = crate::resource::page_mapper() {
        let (to, from, len) = batch.ops[i];
        if !map(to, from, len) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a page mapping failed (applied off the title's call: VICTORY_ASYNC_PAGE_MAPPINGS=1)"));
        }
    }
    APPLYING_HERE.set(before);
    batch.op_state[i].store(OP_DONE, Ordering::Release);
    drop(lock(&FINISHED.0));
    FINISHED.1.notify_all();
    true
}

/// Who applied a batch whole, for the counts.
#[derive(Clone, Copy)]
enum By {
    /// [`worker`], its line having got to it.
    Line,
    /// The thread of a present that forced the lines.
    Forced,
}

/// Since the last queue line ([`STATS`]' places).
const S_LINE: usize = 0;
const S_FORCED: usize = 1;
const S_FAULTS: usize = 2;
const S_MAPPINGS: usize = 3;
const S_PAGES: usize = 4;
const S_APPLY_US: usize = 5;
const S_LAG_US: usize = 6;
const S_EARLY: usize = 7;
const S_WAIT_US: usize = 8;
const S_CALLS: usize = 9;
const S_CALL_US: usize = 10;

/// Since the last queue line: batches applied whole by [`worker`] as their
/// line got to them, and at a present that forced the lines; accesses of
/// the title's that faulted on a target and applied its mappings, and
/// mappings of the title's own that applied batches' mappings under them
/// first; the pages of the batches applied whole, the microseconds that
/// took, and the microseconds from their call to their end; pages applied
/// before their line got to them; microseconds title threads spent on
/// batches (forced, faulted, before their own mappings); the title's calls
/// and its microseconds in them.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// Applies `batch` whole, taken with [`claim`]: its ops not applied yet,
/// in order; those another thread applies are waited for.
fn apply(batch: &Arc<Batch>, by: By) {
    let started = Instant::now();
    // What flights in the air write there goes to the pages shown now.
    crate::host::land_writes_in(&batch.targets);
    for i in 0..batch.ops.len() {
        run_op(batch, i);
    }
    finish(batch, by, started);
}

fn finish(batch: &Arc<Batch>, by: By, started: Instant) {
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
    STATS[match by {
        By::Line => S_LINE,
        By::Forced => S_FORCED,
    }]
    .fetch_add(1, Ordering::Relaxed);
    STATS[S_PAGES].fetch_add(batch.pages, Ordering::Relaxed);
    STATS[S_APPLY_US].fetch_add(us, Ordering::Relaxed);
    STATS[S_LAG_US].fetch_add(lag, Ordering::Relaxed);
    if matches!(by, By::Line) {
        crate::framelog::add(crate::framelog::T::PageMapBatches, 1);
        crate::framelog::add(crate::framelog::T::PageMapPages, batch.pages);
        crate::framelog::add(crate::framelog::T::PageMapApplyUs, us);
        crate::framelog::add(crate::framelog::T::PageMapLagUs, lag);
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
        "page mappings (VICTORY_ASYNC_PAGE_MAPPINGS=1): {batches} batches of {} pages applied, {} by our thread as their line got to them, {} at a present that forced the lines; {:.1} ms applying them, {:.1} ms from the call to the end on average; {} accesses of the title's that faulted on a target and {} mappings of its own applied {} pages before their line got to them; the title spent {:.1} ms on batches and {:.1} ms in {} calls",
        n[S_PAGES],
        n[S_LINE],
        n[S_FORCED],
        n[S_APPLY_US] as f64 / 1e3,
        n[S_LAG_US] as f64 / 1e3 / batches as f64,
        n[S_FAULTS],
        n[S_MAPPINGS],
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

/// A title thread spent the time `since` on batches.
fn waited(since: Instant) {
    let us = since.elapsed().as_micros() as u64;
    STATS[S_WAIT_US].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::PageMapWaitUs, us);
}

/// How long a thread waits for mappings another thread is applying before
/// it says so and goes on: that thread needs nothing of the waiting one's,
/// so only a fault of ours would keep it that long.
const WAIT_MOST: Duration = Duration::from_secs(10);

/// Waits until `done`, [`WAIT_MOST`] at the longest (then says `what` was
/// not done): whether it was.
fn wait(done: impl Fn() -> bool, what: &str) -> bool {
    if done() {
        return true;
    }
    let until = Instant::now() + WAIT_MOST;
    let mut finished = lock(&FINISHED.0);
    while !done() {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            tracing::warn!("{what} was waited for {WAIT_MOST:?} and is not done: going on without it");
            return false;
        }
        finished = FINISHED.1.wait_timeout(finished, left).unwrap_or_else(|e| e.into_inner()).0;
    }
    true
}

/// Applies `batch` whole on this thread, or waits until whoever took it is
/// done.
fn here(batch: &Arc<Batch>, by: By) {
    if claim(batch) {
        apply(batch, by);
    } else {
        wait(|| batch.done(), "a page-mapping batch");
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

/// The mappings not applied yet that must be before an access to title
/// memory `[at, at + len)`, or (`sources`) before the title changes what
/// it shows: the ops over it (or, `sources`, mapping from it), and before
/// each of those the earlier ops over its target or its source (they
/// change what it maps over, or what it maps from, before it). By batch in
/// call order, each batch's ops in order.
fn needed(at: usize, len: usize, sources: bool) -> Vec<(Arc<Batch>, Vec<usize>)> {
    let first = (at, at.saturating_add(len.max(1)));
    let unapplied = lock(&UNAPPLIED);
    let mut wanted = vec![first];
    let mut out = Vec::new();
    for b in unapplied.iter().rev() {
        if !b.targets.iter().any(|&t| meets(&wanted, t)) && !(sources && meets(&b.sources, first)) {
            continue;
        }
        let mut ops = Vec::new();
        for i in (0..b.ops.len()).rev() {
            if b.op_state[i].load(Ordering::Acquire) == OP_DONE {
                continue;
            }
            let (target, source) = op_ranges(&b.ops[i]);
            if meets(&wanted, target) || (sources && source.is_some_and(|s| meets(&[first], s))) {
                wanted.push(target);
                wanted.extend(source);
                ops.push(i);
            }
        }
        if !ops.is_empty() {
            ops.reverse();
            out.push((b.clone(), ops));
        }
    }
    out.reverse();
    out
}

/// Applies `needed` on this thread ([`run_op`]), flights writing there
/// landed first: the pages applied before their line got to them.
fn apply_needed(needed: &[(Arc<Batch>, Vec<usize>)]) -> u64 {
    let mut early = 0;
    for (b, ops) in needed {
        let targets: Vec<(usize, usize)> = ops.iter().map(|&i| op_ranges(&b.ops[i]).0).collect();
        crate::host::land_writes_in(&joined(targets));
        let pending = b.state.load(Ordering::Acquire) == PENDING;
        for &i in ops {
            if run_op(b, i) && pending {
                early += b.ops[i].2.div_ceil(PAGE);
            }
        }
    }
    STATS[S_EARLY].fetch_add(early, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::PageMapEarly, early);
    early
}

/// Before the title maps or unmaps physical pages itself at title memory
/// `[at, at + len)` (`xmem`'s `XMemMapPhysicalPages`, without its locks):
/// the batches' mappings not applied yet over that memory, or mapping from
/// it, are applied first, on this thread, so that mappings change in the
/// order they were asked for. Nothing to do (one atomic look) while no
/// batch waits.
pub fn title_memory_mapping(at: usize, len: usize) {
    if HOW_MANY.load(Ordering::Acquire) == 0 {
        return;
    }
    let needed = needed(at, len, true);
    if needed.is_empty() {
        return;
    }
    let waiting = Instant::now();
    apply_needed(&needed);
    STATS[S_MAPPINGS].fetch_add(1, Ordering::Relaxed);
    waited(waiting);
}

/// The thread that applies batches whole, in the order their lines got to
/// them.
fn worker() {
    crate::framelog::keep_on_fast_cores("victory-remap");
    loop {
        let batch = {
            let mut work = lock(&WORK);
            loop {
                if let Some(b) = work.pop_front() {
                    break b;
                }
                work = WORK_MOVED.wait(work).unwrap_or_else(|e| e.into_inner());
            }
        };
        if claim(&batch) {
            apply(&batch, By::Line);
        }
    }
}

/// Whether `at` is committed memory (a guard or another protection
/// faulted there: not ours to help).
fn committed(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT
}

/// An access violation at `at` by the instruction at `code`: if it is the
/// title's own code, on a thread that holds no reading of title memory in
/// place, and mappings not applied yet are over that page (it shows
/// nothing yet, or is being remapped), this thread applies them now, out
/// of their batches' turn; true if the access may be tried again.
///
/// The thread waits for nothing but ops another thread is applying right
/// now, and that one waits for nothing of this one's: in title code a
/// thread holds no lock of ours, and the remapping's wait for readers
/// (`crate::watch::title_memory_remapping`) cannot be a wait for this
/// thread, which holds no reading. Faults in our code (a copy in place
/// that takes a page not there for an answer, `crate::plaincopy`; a write
/// of DirectStorage's) are left as they were.
fn fault_at(at: usize, code: usize) -> bool {
    if HOW_MANY.load(Ordering::Acquire) == 0 || APPLYING_HERE.get() || crate::watch::reading_here() || !in_title_code(code) {
        return false;
    }
    let page = at - at % PAGE as usize;
    // A first look that allocates nothing: most faults here meet no batch.
    if !lock(&UNAPPLIED).iter().any(|b| meets(&b.targets, (page, page + PAGE as usize))) || committed(at) {
        return false;
    }
    let needed = needed(page, PAGE as usize, false);
    if needed.is_empty() {
        return false;
    }
    let waiting = Instant::now();
    apply_needed(&needed);
    STATS[S_FAULTS].fetch_add(1, Ordering::Relaxed);
    waited(waiting);
    true
}

/// Registers the handler that takes access violations to [`fault_at`],
/// ahead of the handlers registered before it (the guards', the in-place
/// copies', the crash reporter's): it only acts on faults in title code.
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
        // SAFETY: the system passes the exception being dispatched and the
        // context it was raised in.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &*(*info).ContextRecord) };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        #[cfg(target_arch = "x86_64")]
        let code = context.Rip as usize;
        #[cfg(not(target_arch = "x86_64"))]
        let code = {
            let _ = context;
            0
        };
        if fault_at(r.ExceptionInformation[1], code) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
    }
    // SAFETY: the handler only looks at our batches, and applies them on
    // a thread in title code.
    unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The page mapper of the tests: notes each call, with the value of
    /// the fence watched for its target then ([`watch`]); at targets
    /// [`REAL`] names it does what the runtime's does around a mapping
    /// (`xmem`'s `map_alias`): waits for reads in place there to end, then
    /// commits and marks the page.
    pub(crate) fn mapper(to: u64, from: u64, len: u64) -> bool {
        let fence = lock(&FENCES).iter().find(|f| f.0 == to).map(|f| f.1);
        // SAFETY: a fence of the test's, alive for the whole test binary.
        let value = fence.map(|f| crate::fence::completed(f as crate::object::This));
        lock(&CALLS).push((to, from, len, value));
        if lock(&REAL).contains(&to) {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
            let _changing = crate::watch::title_memory_remapping(to as usize, len as usize);
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

    /// The test binary plays the title's code, the thread of ours runs,
    /// the fault handler is in place, and a fresh page of ours is reserved
    /// for the mapper to map (kept: given back, its addresses go to the
    /// guards' tests, whose state by address does not like that).
    fn real_page() -> u64 {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        use_mapper();
        // SAFETY: our own image's headers: the PE offset, then SizeOfImage.
        let (base, size) = unsafe {
            let base = GetModuleHandleW(std::ptr::null()) as usize;
            let nt = base + ((base + 0x3c) as *const u32).read() as usize;
            (base, ((nt + 0x50) as *const u32).read() as usize)
        };
        title_code(base, size);
        start();
        // SAFETY: a plain reservation.
        let at = unsafe { VirtualAlloc(std::ptr::null(), PAGE as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(at, 0);
        lock(&REAL).push(at);
        at
    }

    /// An address in the code [`real_page`] names the title's.
    fn title_rip() -> usize {
        let code: fn(usize, usize) -> bool = fault_at;
        code as usize
    }

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

    /// An access, or a mapping of the title's, needs the mappings over its
    /// memory and, before them, those that change what they map from.
    #[test]
    fn what_an_access_needs_is_found_in_call_order() {
        let base = BASE + 0x1000 * PAGE;
        let at = |k: u64| base + k * PAGE;
        // a maps over the pages b maps from; b maps T; c maps elsewhere; d
        // maps elsewhere, then T again.
        let a = batch(vec![(at(10), 0, PAGE)]);
        let b = batch(vec![(at(0), at(10), PAGE)]);
        let c = batch(vec![(at(50), at(60), PAGE)]);
        let d = batch(vec![(at(70), 0, PAGE), (at(0), at(20), PAGE)]);
        let names = |v: Vec<(Arc<Batch>, Vec<usize>)>| -> Vec<(&str, Vec<usize>)> {
            v.into_iter()
                .map(|(x, ops)| ([(&a, "a"), (&b, "b"), (&c, "c"), (&d, "d")].iter().find(|(y, _)| Arc::ptr_eq(&x, y)).map_or("?", |p| p.1), ops))
                .collect()
        };
        assert_eq!(names(needed(at(0) as usize + 5, 1, false)), [("a", vec![0]), ("b", vec![0]), ("d", vec![1])], "d's other page waits");
        assert_eq!(names(needed(at(50) as usize, PAGE as usize, false)), [("c", vec![0])]);
        assert!(needed(at(60) as usize, PAGE as usize, false).is_empty(), "a source is not an access's");
        assert_eq!(names(needed(at(60) as usize, PAGE as usize, true)), [("c", vec![0])], "the title maps over c's source");
        assert_eq!(names(needed(at(20) as usize, PAGE as usize, true)), [("a", vec![0]), ("b", vec![0]), ("d", vec![1])]);
        use_mapper();
        for x in [&a, &b, &c, &d] {
            run(x);
        }
        assert!(needed(at(0) as usize, PAGE as usize, true).is_empty(), "done: nothing needed");
    }

    /// The title maps memory a batch maps from: that mapping is applied
    /// first, on the title's thread, ahead of its line; the batch's other
    /// page waits for the line, and the batch applied whole maps each page
    /// once.
    #[test]
    fn the_titles_own_mapping_applies_the_mappings_under_it_first() {
        use_mapper();
        let base = BASE + 0x2000 * PAGE;
        let b = batch(vec![(base, base + 0x40 * PAGE, PAGE), (base + 5 * PAGE, 0, PAGE)]);
        title_memory_mapping((base + 0x40 * PAGE) as usize + 100, 10);
        assert_eq!(calls_at(base), [None], "applied before the title's mapping");
        assert!(calls_at(base + 5 * PAGE).is_empty() && !b.done(), "the rest waits for its line");
        run(&b);
        assert!(b.done());
        assert_eq!((calls_at(base).len(), calls_at(base + 5 * PAGE).len()), (1, 1), "each page once");
    }

    /// A read in title code of a target that shows nothing yet: the
    /// faulting thread applies the page's mapping itself and reads it,
    /// even with our thread stuck behind a reading of another page.
    #[test]
    fn a_title_read_of_a_target_not_mapped_yet_maps_it_on_its_own_thread() {
        let (stuck, at) = (real_page(), real_page());
        // Another thread holds a reading of `stuck`: our thread's remapping
        // of it waits.
        let (hold, held) = (std::sync::mpsc::channel::<()>(), std::sync::mpsc::channel::<()>());
        let reader = std::thread::spawn(move || {
            let _reading = crate::watch::title_memory_reading(stuck as usize, PAGE as usize);
            held.0.send(()).unwrap();
            let _ = hold.1.recv();
        });
        held.1.recv().unwrap();
        let s = batch(vec![(stuck, 0x1000, PAGE)]);
        hand(&s);
        let b = batch(vec![(at, 0x1000, PAGE)]);
        assert_eq!(state(&b), PENDING, "its line has not got to it");
        let started = Instant::now();
        // SAFETY: reserved, not committed: the read faults, and the
        // handler has this thread map it through the batch.
        let read = unsafe { std::ptr::read_volatile(at as *const u8) };
        assert_eq!(read, 0x5a, "the read saw the page the batch maps");
        assert!(started.elapsed() < Duration::from_secs(2), "no wait for our thread: {:?}", started.elapsed());
        assert!(!s.done(), "our thread is still stuck behind the reading");
        assert!(!b.done(), "applied for its page only; the batch waits for its line");
        run(&b);
        assert_eq!(calls_at(at), [None], "mapped once");
        hold.0.send(()).unwrap();
        reader.join().unwrap();
        assert!(wait(|| s.done(), "the stuck batch"), "our thread goes on once the reading ends");
    }

    /// Run AH's cycle: a thread of ours holds a reading of title memory in
    /// place and copies a page of it with a copy that takes a page not there
    /// for an answer (`crate::plaincopy`); the page is a target of a batch
    /// being applied, whose remapping waits for that reading. The fault is
    /// not ours to help: the copy says "not there" at once, and the batch
    /// goes on once the reading ends.
    #[test]
    fn a_copy_of_ours_under_a_reading_never_waits_for_a_batch() {
        let at = real_page();
        let mut buffer = [0u8; 16];
        // The in-place copies' handler, then ours ahead of it (as in a run:
        // the copies' handler is put in place first).
        assert!(crate::plaincopy::guarded_copy(buffer.as_mut_ptr(), [7u8; 16].as_ptr(), 16));
        install();
        let reading = crate::watch::title_memory_reading(at as usize, PAGE as usize);
        let b = batch(vec![(at, 0x1000, PAGE)]);
        hand(&b);
        let started = Instant::now();
        let copied = crate::plaincopy::guarded_copy(buffer.as_mut_ptr(), at as *const u8, 16);
        assert!(!copied, "the page is not there yet");
        assert!(started.elapsed() < Duration::from_secs(2), "the copy waited: {:?}", started.elapsed());
        assert!(!fault_at(at as usize, title_rip()), "under a reading nothing is applied here");
        assert!(!b.done(), "its remapping waits for the reading");
        drop(reading);
        assert!(wait(|| b.done(), "the batch"), "applied once the reading ends");
        assert!(crate::plaincopy::guarded_copy(buffer.as_mut_ptr(), at as *const u8, 16));
        assert_eq!(buffer[0], 0x5a);
    }

    /// A fault outside the title's code is never helped.
    #[test]
    fn faults_outside_title_code_are_left_alone() {
        let at = real_page();
        let b = batch(vec![(at, 0x1000, PAGE)]);
        assert!(!fault_at(at as usize, 0x1000), "not title code");
        assert!(calls_at(at).is_empty());
        assert!(fault_at(at as usize, title_rip()), "title code, no reading: applied here");
        assert_eq!(calls_at(at), [None]);
        run(&b);
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

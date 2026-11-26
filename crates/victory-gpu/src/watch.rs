//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never". Taken under either lock (an
    /// answer under the shared one takes a number too: [`writes_shared`]).
    asked: AtomicU64,
    /// Page number → the asking that last found it written.
    written: Askings,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Ahead,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

impl Journal {
    /// A new asking's number, higher than every one before.
    fn next_asking(&self) -> u64 {
        self.asked.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
    }
}

/// Pages to a chunk of [`Askings`]: 256 MiB of title memory, 32 KiB of
/// askings.
const ASK_SHIFT: u32 = 12;
const ASK_CHUNK: usize = 1 << ASK_SHIFT;

/// The journal's pages: for each, the asking that last found it written
/// (0: none; every asking is numbered from 1), in chunks of [`ASK_CHUNK`]
/// pages made when a page of theirs is first noted (and kept), in chunk
/// order, with the highest asking of each 64 pages beside them (at least
/// as high: one noted lower later leaves it). The pages of a range written
/// after an asking are found without looking into groups of 64 that hold
/// none ([`Askings::after`]).
///
/// It was a `BTreeMap`, whose range was read whole into a vector at every
/// asking: 3.4 ms a frame of the submitting thread in run AM (the walk of
/// the tree, the vector, its freeing), for copies that compared nearly all
/// of it with their own askings and found it older.
struct Askings {
    chunks: Vec<AskChunk>,
}

struct AskChunk {
    /// Its number: pages `[at * ASK_CHUNK, (at + 1) * ASK_CHUNK)`.
    at: usize,
    askings: Box<[u64; ASK_CHUNK]>,
    /// Per 64 pages, an asking no lower than any of theirs.
    most: [u64; ASK_CHUNK / 64],
}

impl Askings {
    const fn new() -> Askings {
        Askings { chunks: Vec::new() }
    }

    /// The asking that last found `page` written; 0 if none.
    #[cfg(test)]
    fn get(&self, page: usize) -> u64 {
        match self.chunks.binary_search_by_key(&(page >> ASK_SHIFT), |c| c.at) {
            Ok(i) => self.chunks[i].askings[page & (ASK_CHUNK - 1)],
            Err(_) => 0,
        }
    }

    /// Notes `page` written at `asking`; the asking noted before (0: none).
    fn set(&mut self, page: usize, asking: u64) -> u64 {
        let i = match self.chunks.binary_search_by_key(&(page >> ASK_SHIFT), |c| c.at) {
            Ok(i) => i,
            Err(i) => {
                let chunk = AskChunk { at: page >> ASK_SHIFT, askings: Box::new([0; ASK_CHUNK]), most: [0; ASK_CHUNK / 64] };
                self.chunks.insert(i, chunk);
                i
            }
        };
        let c = &mut self.chunks[i];
        let o = page & (ASK_CHUNK - 1);
        c.most[o / 64] = c.most[o / 64].max(asking);
        std::mem::replace(&mut c.askings[o], asking)
    }

    /// The pages of `[first, last)` noted written after asking `since`, with
    /// the asking that last found each, in page order (`since` 0: all that
    /// were ever noted).
    fn after(&self, first: usize, last: usize, since: u64) -> Vec<(usize, u64)> {
        let mut found = Vec::new();
        self.after_into(first, last, since, &mut found);
        found
    }

    /// [`Askings::after`], added to `found` (the refreshes' own buffer,
    /// [`writes_after_into`]: run BA's play had 0.7 ms a frame of the
    /// submitting thread growing a new vector for each answer).
    fn after_into(&self, first: usize, last: usize, since: u64, found: &mut Vec<(usize, u64)>) {
        if first >= last {
            return;
        }
        let (lo_chunk, hi_chunk) = (first >> ASK_SHIFT, (last - 1) >> ASK_SHIFT);
        let start = self.chunks.partition_point(|c| c.at < lo_chunk);
        for c in self.chunks[start..].iter().take_while(|c| c.at <= hi_chunk) {
            let base = c.at << ASK_SHIFT;
            let (lo, hi) = (first.max(base) - base, last.min(base + ASK_CHUNK) - base);
            for group in lo / 64..hi.div_ceil(64) {
                if c.most[group] <= since {
                    continue;
                }
                for o in (group * 64).max(lo)..(group * 64 + 64).min(hi) {
                    let asking = c.askings[o];
                    if asking > since {
                        found.push((base + o, asking));
                    }
                }
            }
        }
    }

    /// Tests: forgets pages `[first, last)`.
    #[cfg(test)]
    fn forget(&mut self, first: usize, last: usize) {
        for page in first..last {
            if let Ok(i) = self.chunks.binary_search_by_key(&(page >> ASK_SHIFT), |c| c.at) {
                self.chunks[i].askings[page & (ASK_CHUNK - 1)] = 0;
            }
        }
    }
}

/// Page ranges asked ahead ([`Journal::ahead`]), sorted by first page, each
/// with the farthest end of it and those before: whether one holds a range
/// is a binary search. A submission's refreshes ask it some 10,000 times a
/// frame, each a walk of every range before. (After VIctory-7900XTX
/// c83e54c.)
struct Ahead {
    ranges: Vec<(usize, usize)>,
    /// `most[i]`: the farthest end of `ranges[..=i]`.
    most: Vec<usize>,
}

impl Ahead {
    const fn new() -> Ahead {
        Ahead { ranges: Vec::new(), most: Vec::new() }
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.most.clear();
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn push(&mut self, range: (usize, usize)) {
        let at = self.ranges.partition_point(|r| r.0 <= range.0);
        self.ranges.insert(at, range);
        self.most.truncate(at);
        let mut most = self.most.last().copied().unwrap_or(0);
        for r in &self.ranges[at..] {
            most = most.max(r.1);
            self.most.push(most);
        }
    }

    fn extend_from_slice(&mut self, ranges: &[(usize, usize)]) {
        for &range in ranges {
            self.push(range);
        }
    }

    /// Whether a range holds pages `[first, last)`.
    fn holds(&self, first: usize, last: usize) -> bool {
        let n = self.ranges.partition_point(|r| r.0 <= first);
        n > 0 && self.most[n - 1] >= last
    }
}

/// Behind a lock that askings which change nothing share ([`writes_shared`]);
/// every other use holds it alone ([`journal`]), as when it was a mutex.
static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    asked: AtomicU64::new(0),
    written: Askings::new(),
    found: Vec::new(),
    ahead: Ahead::new(),
    ahead_asked: 0,
    kept: false,
});

/// The journal, held alone.
fn journal() -> std::sync::RwLockWriteGuard<'static, Journal> {
    JOURNAL.write().unwrap_or_else(|e| e.into_inner())
}

/// The journal, held with others that only look ([`writes_shared`]).
fn journal_shared() -> std::sync::RwLockReadGuard<'static, Journal> {
    JOURNAL.read().unwrap_or_else(|e| e.into_inner())
}

/// Tests: what the journal noted of `[start, end)` goes, before the test
/// lets its memory go. Its pages would stay "written" for the next test
/// whose memory the system puts at the same addresses.
#[cfg(test)]
pub(crate) fn forget_written(start: usize, end: usize) {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    journal().written.forget(first, last);
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// [`AHEAD_GAP`] in bytes: ranges of one copy closer than this are one
/// refresh with `VICTORY_ASK_EXACT` ([`ask_exact`]), and so lie in one
/// asking.
pub(crate) const AHEAD_GAP_BYTES: usize = AHEAD_GAP * PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Whether a submission's one asking ([`ask_once`]) is about the ranges
/// its lists refresh, not about the span of each copy's refreshed ranges
/// (`VICTORY_ASK_EXACT=1`; `alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log; off by default). Read once a
/// submission (`crate::host`): its refreshes and the look under the flights
/// go by what it says there too.
///
/// A kernel asking walks every page of its range (by hand, 2026-11-26, a
/// test process of 1 GiB write-watched: 0.75 us a MB alone, about 1.4 with
/// 24 threads of the process running beside, so 0.8-1.5 ms for 1 GiB,
/// whatever was written; a failed asking 0.3-0.5 us, and some 10 us more
/// when it resets written pages, the shootdown of TLBs). Run BA's play
/// spent 3.7 ms a frame of the submitting thread in the askings of
/// `ask_once` (9.6 submissions a frame) and run BL had 1.2 askings a frame
/// of a millisecond or more (2.4 ms a frame): runs of hundreds of MB. A
/// copy of a big buffer whose lists read pieces far apart was asked about
/// from its first piece to its last. With the switch the ranges themselves
/// are asked about (those of a copy closer than [`AHEAD_GAP`] together, as
/// before), and a copy's refreshes of one flush are split where its ranges
/// lie farther apart (`crate::host`'s `Batch`), so each still lies in one
/// asking and is answered from the journal. What the lists do not read is
/// neither asked about nor copied. `ahead_line` says what was asked.
pub(crate) fn ask_exact() -> bool {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_ASK_EXACT").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let on = (crate::host::now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a submission's one write-watch asking from now (VICTORY_ASK_EXACT=alt): {}",
                    if on { "EXACT, about the ranges refreshed" } else { "off, about each copy's span as usual" }
                );
            }
            on
        }
    }
}

/// Since the last frame line ([`ahead_line`]), of the askings ahead
/// ([`ask_ahead`], [`ask_once`]): runs of two ranges or more asked
/// together, the pages they spanned (what the kernel walks), the pages of
/// the ranges in them; ranges asked alone, their pages; the microseconds
/// of `ask_once`, and its submissions; of the submissions sampled
/// ([`note_refreshed`]), the pages of the ranges their lists refresh, and
/// how many.
static AHEAD: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// The pages of the ranges a submission's lists refresh, all copies
/// together (sampled by `crate::host`): what an exact asking would cover.
pub(crate) fn note_refreshed(pages: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    AHEAD[7].fetch_add(pages, Relaxed);
    AHEAD[8].fetch_add(1, Relaxed);
}

/// With the frame line: what [`AHEAD`] counted, per submission.
pub(crate) fn ahead_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let n = AHEAD.each_ref().map(|a| a.swap(0, Relaxed) as f64);
    let (subs, sampled) = (n[6].max(1.0), n[8].max(1.0));
    let mb = |pages: f64| pages * (PAGE >> 10) as f64 / 1024.0;
    tracing::info!(
        "write-watch askings ahead per submission ({:.1} submissions a frame; VICTORY_ASK_EXACT {}): {:.1} runs over {:.0} MB \
         (what the kernel walks), the ranges asked in them {:.0} MB; {:.1} ranges asked alone, {:.0} MB; {:.0} us in all; \
         the ranges the lists refresh, all copies together, {:.0} MB (one submission in 8)",
        n[6] / frames,
        if ask_exact() { "on" } else { "off" },
        n[0] / subs,
        mb(n[1] / subs),
        mb(n[2] / subs),
        n[3] / subs,
        mb(n[4] / subs),
        n[5] / subs,
        mb(n[7] / sampled)
    );
}

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = journal();
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    // (The guards looked at once for all the pieces, not once a piece:
    // run AM, 0.3 ms a frame taking their lock.)
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    if !views_asked() {
        let g = guards_shared();
        pieces.retain(|&(first, _)| !known_view_in(&g, first * PAGE));
    }
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.ahead_asked = j.next_asking();
    j.ahead.clear();
    let (runs, spanned, held) = ahead_runs(&pieces);
    {
        use std::sync::atomic::Ordering::Relaxed;
        AHEAD[0].fetch_add(runs.iter().filter(|r| r.1 - r.0 >= 2).count() as u64, Relaxed);
        AHEAD[1].fetch_add(spanned as u64, Relaxed);
        AHEAD[2].fetch_add(held as u64, Relaxed);
    }
    if ask_wide() && runs.iter().filter(|r| r.1 - r.0 >= 2).count() >= 2 {
        let answers = crate::shadow::spread_own(runs.len(), &|r| {
            let (i, k) = runs[r];
            let mut asked = Vec::new();
            let pages = KERNEL_SCRATCH.with(|scratch| {
                let mut scratch = scratch.take();
                scratch.clear();
                kernel_run(&pieces[i..k], &mut asked, &mut scratch);
                let pages = scratch.clone();
                KERNEL_SCRATCH.set(scratch);
                pages
            });
            (asked, pages)
        });
        for (asked, pages) in answers {
            note_ahead(j, &asked, &pages);
        }
        return;
    }
    for (i, k) in runs {
        ask_run(j, &pieces[i..k]);
    }
}

/// The runs of `pieces` (page ranges, in order) that [`ask_ahead`] asks
/// about together: `[i, k)` of `pieces` each, pieces closer than
/// [`AHEAD_GAP`] in one, none spanning more than [`AHEAD_MAX`]; and of the
/// runs of two pieces or more (the ones it asks), the pages they span (what
/// the kernel walks) and the pages of their pieces.
fn ahead_runs(pieces: &[(usize, usize)]) -> (Vec<(usize, usize)>, usize, usize) {
    let mut runs = Vec::new();
    let (mut spanned, mut held) = (0, 0);
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        // The pieces' union so far: whole up to `covered`.
        let (mut union, mut covered) = (last - first, last);
        while k < pieces.len() && pieces[k].0 <= last + AHEAD_GAP && last.max(pieces[k].1) - first <= AHEAD_MAX {
            union += pieces[k].1.saturating_sub(pieces[k].0.max(covered));
            covered = covered.max(pieces[k].1);
            last = last.max(pieces[k].1);
            k += 1;
        }
        if k - i >= 2 {
            (spanned, held) = (spanned + (last - first), held + union);
        }
        runs.push((i, k));
        i = k;
    }
    (runs, spanned, held)
}

/// With [`ask_exact`]: whether a copy's range `[address, end)` lies too far
/// from its pending refresh `[from, to)` to be merged into it (`crate::host`'s
/// `Batch`): farther than [`AHEAD_GAP_BYTES`] on either side. Merged only
/// closer than that, a refresh lies in one run of [`ahead_runs`] made of the
/// submission's ranges.
pub(crate) fn apart(from: usize, to: usize, address: usize, end: usize) -> bool {
    address > to.saturating_add(AHEAD_GAP_BYTES) || end.saturating_add(AHEAD_GAP_BYTES) < from
}

thread_local! {
    /// The kernel's answers to [`kernel_run`] on a kept thread
    /// ([`ask_wide`]), kept from one asking to the next.
    static KERNEL_SCRATCH: std::cell::Cell<Vec<usize>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = journal();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let started = std::time::Instant::now();
    ask_once_here(ranges);
    AHEAD[5].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    AHEAD[6].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`ask_once`], untimed.
fn ask_once_here(ranges: &[(usize, usize)]) {
    ask_once_end();
    ask_ahead(ranges);
    let mut j = journal();
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.ahead_asked = j.next_asking();
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    if ask_wide() {
        let mut lone = Vec::new();
        for &(start, end) in ranges {
            if start >= end {
                continue;
            }
            if known_view(start) {
                #[cfg(feature = "diagnostics")]
                AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            let (first, last) = (start / PAGE, end.div_ceil(PAGE));
            if !j.ahead.holds(first, last) {
                lone.push((first, last));
            }
        }
        lone.sort_unstable();
        lone.dedup();
        let answers = crate::shadow::spread_own(lone.len(), &|r| {
            let (first, last) = lone[r];
            let mut pages = Vec::new();
            kernel_pages(first, last, &mut pages).then_some(pages)
        });
        for (&range, pages) in lone.iter().zip(answers) {
            if let Some(pages) = pages {
                note_ahead(j, &[range], &pages);
            }
        }
        j.kept = !j.ahead.is_empty();
        return;
    }
    // View memory the guards hold is not asked about: the asking can only
    // fail there, one failed kernel call for each such copy at every
    // submission, and `writes` answers it from the guards (after
    // triippywhale cc368c7c). The guards are looked at once for all the
    // ranges, as `ask_ahead` does (`VICTORY_ASK_EXACT` gives thousands).
    let lone: Vec<(usize, usize)> = {
        let g = (!views_asked()).then(guards_shared);
        ranges
            .iter()
            .filter(|&&(start, end)| start < end)
            .filter(|&&(start, _)| {
                let known = g.as_ref().is_some_and(|g| known_view_in(g, start));
                #[cfg(feature = "diagnostics")]
                if known {
                    AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                !known
            })
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .collect()
    };
    for (first, last) in lone {
        if !j.ahead.holds(first, last) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
            AHEAD[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            AHEAD[4].fetch_add((last - first) as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = journal();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = journal();
    if !j.kept || len == 0 {
        return;
    }
    let now = j.next_asking();
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.set(page, now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = journal();
    let j = &mut *j;
    let now = j.next_asking();
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.set(page, now);
    for &other in others {
        if other != page {
            j.written.set(other, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    let (mut asked, mut pages) = (Vec::new(), std::mem::take(&mut j.found));
    pages.clear();
    kernel_run(pieces, &mut asked, &mut pages);
    note_ahead(j, &asked, &pages);
    j.found = pages;
}

/// [`ask_run`]'s kernel askings alone: the ranges asked go to `asked`,
/// the pages found written in them to `pages`.
fn kernel_run(pieces: &[(usize, usize)], asked: &mut Vec<(usize, usize)>, pages: &mut Vec<usize>) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if kernel_pages(first, last, pages) {
        asked.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    kernel_run(a, asked, pages);
    kernel_run(b, asked, pages);
}

/// Notes ranges asked ahead and the pages found written in them, under the
/// ahead asking.
fn note_ahead(j: &mut Journal, asked: &[(usize, usize)], pages: &[usize]) {
    j.ahead.extend_from_slice(asked);
    let now = j.ahead_asked;
    for &page in pages {
        j.written.set(page, now);
    }
}

/// Whether the kernel askings of a submission's ranges ahead are made side
/// by side on the kept threads (`VICTORY_ASK_WIDE=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log; off by
/// default): they are of ranges apart from one another and need nothing of
/// the journal, which notes their answers after. One that resets written
/// pages costs 30-70 microseconds; run BA's submitting thread spent 3.7 ms
/// a frame in them.
fn ask_wide() -> bool {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_ASK_WIDE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let on = (crate::host::now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "write-watch askings ahead from now (VICTORY_ASK_WIDE=alt): {}",
                    if on { "WIDE, on the kept threads" } else { "one after the other, as usual" }
                );
            }
            on
        }
    }
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    let mut pages = std::mem::take(&mut j.found);
    pages.clear();
    let asked = kernel_pages(first, last, &mut pages);
    if asked {
        note_ahead(j, &[], &pages);
    }
    j.found = pages;
    asked
}

/// Asks the kernel for the written pages of pages `[first, last)`, and
/// resets them: their numbers go to `pages`. False (and nothing added) if
/// the range is not write-watched memory of one reservation.
fn kernel_pages(first: usize, last: usize, pages: &mut Vec<usize>) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    let at = pages.len();
    // Room for every page, not filled: the kernel writes the entries it
    // gives (a range of 1 GiB has room for 2 MB of them).
    pages.reserve(count);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer has room for `count` entries past `at`.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            pages.as_mut_ptr().add(at).cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    // SAFETY: the kernel wrote the `count` entries past `at`, no more than
    // it had room for.
    unsafe { pages.set_len(at + count) };
    for page in &mut pages[at..] {
        *page /= PAGE;
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal_shared();
    j.ahead
        .holds(first, last)
        .then(|| (j.ahead_asked, j.written.after(first, last, 0)))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let mut written = Vec::new();
    writes_as(start, end, guarded, shared_asks(), 0, &mut written).map(|now| (now, written))
}

/// [`writes`], with the pages last found written at or before asking
/// `since` left out of the list: for a caller whose every page of the
/// range was taken after `since` (or never), to which those pages were
/// not written since it took them. Most refreshes find nothing new: the
/// list of every page ever written in the range, made at each asking and
/// compared page by page with the copy's, was 3.4 ms a frame of the
/// submitting thread in run AM.
pub(crate) fn writes_after(start: usize, end: usize, guarded: bool, since: u64) -> Option<(u64, Vec<(usize, u64)>)> {
    let mut written = Vec::new();
    writes_as(start, end, guarded, shared_asks(), since, &mut written).map(|now| (now, written))
}

/// [`writes_after`] into `written` (emptied first), the caller's buffer
/// kept from one refresh to the next: the asking's number, or `None` as
/// [`writes_after`] says (`written` is then empty).
pub(crate) fn writes_after_into(start: usize, end: usize, guarded: bool, since: u64, written: &mut Vec<(usize, u64)>) -> Option<u64> {
    written.clear();
    writes_as(start, end, guarded, shared_asks(), since, written)
}

/// [`writes_after`] into `written` (left as it is if `None`); with
/// `shared`, an asking that changes nothing is answered under the shared
/// locks ([`writes_shared`]).
fn writes_as(start: usize, end: usize, guarded: bool, shared: bool, since: u64, written: &mut Vec<(usize, u64)>) -> Option<u64> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    // Copies of physical pages there whose source was written are taken
    // again first: their own write watch then says so (`crate::alias`).
    crate::alias::pull(start, end);
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if shared && let Some(now) = writes_shared(first, last, guarded, since, written) {
        return Some(now);
    }
    let mut j = journal();
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.holds(first, last) {
        j.written.after_into(first, last, since, written);
        return Some(j.ahead_asked);
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    let known_view = guarded && known_view(first * PAGE);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    #[cfg(feature = "diagnostics")]
    if failed && !known_view && guarded {
        failed_asking(first * PAGE, last * PAGE);
    }
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.next_asking();
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.set(page, now);
    }
    for page in guarded {
        j.written.set(page, now);
    }
    j.written.after_into(first, last, since, written);
    Some(now)
}

/// Whether an asking that would change nothing is answered under shared
/// locks ([`writes_shared`]) instead of holding the journal alone
/// (`VICTORY_ASK_SHARED`; by default on when `VICTORY_WIDE_REFRESH` is `1`
/// or `alt`, off otherwise; `1` or `0` sets it either way).
///
/// Every asking held the journal's lock alone for its whole length, the
/// guards' check of view memory inside it: run AB's gameplay profile has
/// 20.3 of the 21.2 ms a frame of the submitting thread's refreshes
/// (`update_free`) in [`writes`], 15.5 of them in [`guard`]. So refreshes
/// on several threads (`VICTORY_WIDE_REFRESH`) stood in line there, one
/// asking at a time (run 112 measured them worse than one thread), and
/// the texture threads' askings waited behind the submitting thread's.
/// Most askings find nothing to change: a frame asks about some 150,000
/// blocks of view memory, nearly all clean (run 54), and the ranges the
/// submission's one asking holds are answered from the journal. Those now
/// look side by side; the rest hold the locks alone, as before.
pub(crate) fn shared_asks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_ASK_SHARED").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => matches!(std::env::var("VICTORY_WIDE_REFRESH").as_deref(), Ok("1" | "alt")),
    })
}

/// Since the last frame line ([`shared_line`]): askings answered under the
/// shared locks from the journal, and from the guards; askings that looked
/// there and went on to hold the locks alone.
static SHARED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// With the frame line: what [`writes_shared`] did per frame over `frames`.
pub(crate) fn shared_line(frames: f64) {
    if !shared_asks() {
        return;
    }
    let [journal, guards, alone] = [0, 1, 2].map(|i| SHARED[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "write-watch askings per frame answered side by side (VICTORY_ASK_SHARED): {journal:.0} from the journal, \
         {guards:.0} from the guards (view memory with nothing to guard again); {alone:.0} went on alone"
    );
}

/// [`writes`] of pages `[first, last)` where it would change nothing: the
/// range lies in one asked ahead (the journal's answer, as `writes` gives
/// it), or, with `guarded`, it is view memory the guards hold whose every
/// block is clean or noted absent inside a hole Windows lately said is
/// there ([`unchanged`]): `writes` would guard no block again, find none
/// written, note nothing and ask Windows nothing. `None` where it has more
/// to do; it then does it as before, holding the locks alone.
///
/// Under the journal's lock and the guards', both shared. What changes
/// them holds them alone: an asking that guards blocks again holds both,
/// from its look at the guards to the noting of what it found, and a write
/// the guards take or a remapping holds the guards'. So this answer is the
/// one such an asking would give now, with nothing written; a write the
/// guards take after the look is found by the next asking, as after any.
/// Its number is a new one, taken with the journal held: every write
/// noted before is older, every one noted after (alone) newer.
fn writes_shared(first: usize, last: usize, guarded: bool, since: u64, written: &mut Vec<(usize, u64)>) -> Option<u64> {
    use std::sync::atomic::Ordering::Relaxed;
    let j = journal_shared();
    if j.ahead.holds(first, last) {
        SHARED[0].fetch_add(1, Relaxed);
        j.written.after_into(first, last, since, written);
        return Some(j.ahead_asked);
    }
    // (With the kernel to be asked anyway, as `known_view` has it.)
    if !guarded || views_asked() {
        return None;
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let unchanged = {
        let g = guards_shared();
        known_view_in(&g, first * PAGE) && unchanged(&g, first * PAGE / BLOCK, (last * PAGE).div_ceil(BLOCK))
    };
    if !unchanged {
        SHARED[2].fetch_add(1, Relaxed);
        return None;
    }
    SHARED[1].fetch_add(1, Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        let us = started.elapsed().as_micros() as u64;
        add(C::WatchKnownView, 1);
        add(C::GuardUs, us);
        add(C::Watches, 1);
        add(C::WatchUs, us);
        slow(us);
    }
    let now = j.next_asking();
    j.written.after_into(first, last, since, written);
    Some(now)
}

/// Whether [`guard`] of blocks `[first, last)` would change nothing and
/// find nothing written: each block clean, or noted absent inside a hole
/// Windows lately said is there ([`known_hole`]). Then it guards no block
/// again, opens none, notes none and asks Windows nothing.
fn unchanged(g: &Guards, first: usize, last: usize) -> bool {
    unchanged_with(g, first, last, known_hole)
}

/// [`unchanged`], with `hole` what is known of the hole at a block.
fn unchanged_with(g: &Guards, first: usize, last: usize, hole: impl Fn(&Guards, usize) -> Option<(usize, bool)>) -> bool {
    // As `guard`'s walk: the end of the hole the last absent block lay in.
    let mut hole_until = first;
    let mut at = first;
    loop {
        // Clean blocks pass, and absent ones inside that hole.
        let block = next_open(g, at, hole_until, last);
        if block == last {
            return true;
        }
        // Not clean and not absent: `guard` takes it for written.
        if !g.absent.contains(block) {
            return false;
        }
        // Absent past the hole (`next_open` passed over those inside it).
        // Not known lately: `guard` asks Windows (and notes the answer).
        let Some((end, _)) = hole(g, block) else {
            return false;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HolesKnown, 1);
        hole_until = end;
        at = block + 1;
    }
}

/// The first block of `[at, last)` that [`guard`]'s walk must look at:
/// not clean, nor absent before `hole_until` (inside the hole the walk
/// last found). Read a word of 64 blocks at a time.
fn next_open(g: &Guards, at: usize, hole_until: usize, last: usize) -> usize {
    let inside = hole_until.min(last);
    if at < inside {
        let block = Blocks::first_in_none(&g.clean, Some(&g.absent), at, inside);
        if block < inside {
            return block;
        }
        return g.clean.first_unset(inside, last);
    }
    g.clean.first_unset(at, last)
}

/// [`next_open`] with the quiet blocks ([`hole_bits`]) passed over as
/// well: absent blocks inside a hole lately said are there.
fn next_open_quiet(g: &Guards, at: usize, hole_until: usize, last: usize) -> usize {
    let inside = hole_until.min(last);
    let mut at = at;
    if at < inside {
        let block = Blocks::first_in_none(&g.clean, Some(&g.absent), at, inside);
        if block < inside {
            return block;
        }
        at = inside;
    }
    Blocks::first_in_none(&g.clean, Some(&g.quiet), at, last)
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know (some 200 a frame in the
/// first rooms, each followed by a question to Windows about the memory):
/// what the memory is and what the guards hold of its first block.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2048 != 0 {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let (protected, clean, absent, viewed, aliased) = {
        let g = guards_shared();
        let b = start / BLOCK;
        (g.protected.contains(b), g.clean.contains(b), g.absent.contains(b), g.viewed.contains(b), g.aliased.contains(b))
    };
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: protected {protected} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        is_view(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
/// Whether the guarded block holding `at` is clean: guarded and not
/// written since (a look at the guards' set, nothing asked of Windows).
pub(crate) fn block_clean(at: usize) -> bool {
    guards_shared().clean.contains(at / BLOCK)
}

fn known_view(start: usize) -> bool {
    !views_asked() && known_view_in(&guards_shared(), start)
}

/// `VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`: the kernel is asked about
/// view memory the guards hold ([`known_view`]).
fn views_asked() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// [`known_view`]'s look at the guards `g`, the switches aside. A block
/// left open to rest ([`rest_now`]) was read-only by our hand a moment ago:
/// a view, which the kernel's write watch would be asked about in vain at
/// every check of its rest.
fn known_view_in(g: &Guards, start: usize) -> bool {
    let first = start / BLOCK;
    g.protected.contains(first)
        || g.absent.contains(first)
        || (view_known() && g.viewed.contains(first))
        || (!g.rests.rested.is_empty() && g.rests.rested.contains(first))
}

/// Blocks to a chunk of [`Blocks`]: 2 GiB of title memory, 4 KiB of bits.
const CHUNK_SHIFT: u32 = 15;
const CHUNK: usize = 1 << CHUNK_SHIFT;
const CHUNK_WORDS: usize = CHUNK / 64;

/// A set of block numbers held as bits, in chunks of [`CHUNK`] blocks made
/// when a block of theirs is first set (and kept), in chunk order: what a
/// range holds is read 64 blocks to a word.
///
/// The guards' sets were `BTreeSet`s. Run AM's profile of the submitting
/// thread has 2.3 ms a frame of B-tree walks under [`guard`] (each check
/// read the range's clean blocks off the set into a vector, and looked up
/// the others one by one), 2.0 ms more in its loop over the blocks, and
/// 1.6 ms of lookups in [`known_view`]: a frame checks some 150,000 blocks,
/// nearly all clean.
struct Blocks {
    chunks: Vec<Chunk>,
    /// Blocks set, in all chunks.
    len: usize,
}

struct Chunk {
    /// Its number: blocks `[at * CHUNK, (at + 1) * CHUNK)`.
    at: usize,
    /// How many of them are set.
    set: usize,
    bits: Box<[u64; CHUNK_WORDS]>,
}

/// The bits `[lo, hi)` of a word (`lo < hi <= 64`).
fn word_mask(lo: usize, hi: usize) -> u64 {
    (u64::MAX >> (64 - (hi - lo))) << lo
}

/// The words holding blocks `[lo, hi)` of a chunk (`lo < hi <= CHUNK`), each
/// with the mask of those blocks in it.
fn chunk_words(lo: usize, hi: usize) -> impl Iterator<Item = (usize, u64)> {
    (lo / 64..hi.div_ceil(64)).map(move |w| (w, word_mask(lo.max(w * 64) - w * 64, hi.min(w * 64 + 64) - w * 64)))
}

/// The set bits of a word, lowest first.
struct Ones(u64);

impl Iterator for Ones {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let bit = (self.0 != 0).then(|| self.0.trailing_zeros() as usize);
        self.0 &= self.0.wrapping_sub(1);
        bit
    }
}

impl Blocks {
    const fn new() -> Blocks {
        Blocks { chunks: Vec::new(), len: 0 }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn chunk(&self, at: usize) -> Option<&Chunk> {
        let i = self.chunks.binary_search_by_key(&at, |c| c.at).ok()?;
        Some(&self.chunks[i])
    }

    /// The chunk numbered `at`, made if there is none.
    fn chunk_made(chunks: &mut Vec<Chunk>, at: usize) -> &mut Chunk {
        let i = match chunks.binary_search_by_key(&at, |c| c.at) {
            Ok(i) => i,
            Err(i) => {
                chunks.insert(i, Chunk { at, set: 0, bits: Box::new([0; CHUNK_WORDS]) });
                i
            }
        };
        &mut chunks[i]
    }

    /// The chunks that hold blocks of `[from, to)`.
    fn chunks_in(&self, from: usize, to: usize) -> &[Chunk] {
        if from >= to {
            return &[];
        }
        let (first, last) = (from >> CHUNK_SHIFT, (to - 1) >> CHUNK_SHIFT);
        let start = self.chunks.partition_point(|c| c.at < first);
        let end = start + self.chunks[start..].partition_point(|c| c.at <= last);
        &self.chunks[start..end]
    }

    /// The blocks of `[from, to)` that lie in chunk `c`, as its own (`lo..hi`).
    fn local(c: usize, from: usize, to: usize) -> (usize, usize) {
        let base = c << CHUNK_SHIFT;
        (from.max(base) - base, to.min(base + CHUNK) - base)
    }

    fn contains(&self, block: usize) -> bool {
        let o = block & (CHUNK - 1);
        self.chunk(block >> CHUNK_SHIFT).is_some_and(|c| (c.bits[o / 64] >> (o % 64)) & 1 != 0)
    }

    /// Sets `block`; true if it was not set.
    fn insert(&mut self, block: usize) -> bool {
        let c = Self::chunk_made(&mut self.chunks, block >> CHUNK_SHIFT);
        let (o, bit) = (block & (CHUNK - 1), 1 << (block % 64));
        let new = c.bits[o / 64] & bit == 0;
        c.bits[o / 64] |= bit;
        c.set += usize::from(new);
        self.len += usize::from(new);
        new
    }

    /// Clears `block`; true if it was set.
    fn remove(&mut self, block: usize) -> bool {
        let Ok(i) = self.chunks.binary_search_by_key(&(block >> CHUNK_SHIFT), |c| c.at) else {
            return false;
        };
        let c = &mut self.chunks[i];
        let (o, bit) = (block & (CHUNK - 1), 1 << (block % 64));
        let was = c.bits[o / 64] & bit != 0;
        c.bits[o / 64] &= !bit;
        c.set -= usize::from(was);
        self.len -= usize::from(was);
        was
    }

    /// Sets blocks `[from, to)`.
    fn insert_range(&mut self, from: usize, to: usize) {
        let mut at = from;
        while at < to {
            let n = at >> CHUNK_SHIFT;
            let (lo, hi) = Self::local(n, at, to);
            let c = Self::chunk_made(&mut self.chunks, n);
            let mut added = 0;
            for (w, mask) in chunk_words(lo, hi) {
                added += (mask & !c.bits[w]).count_ones() as usize;
                c.bits[w] |= mask;
            }
            c.set += added;
            self.len += added;
            at = (n + 1) << CHUNK_SHIFT;
        }
    }

    /// Sets the blocks of `[from, to)` that `other` holds; how many were
    /// not set before.
    fn insert_and(&mut self, other: &Blocks, from: usize, to: usize) -> usize {
        let mut added = 0;
        for c in other.chunks_in(from, to) {
            if c.set == 0 {
                continue;
            }
            let (lo, hi) = Self::local(c.at, from, to);
            let mine = Self::chunk_made(&mut self.chunks, c.at);
            let mut n = 0;
            for (w, mask) in chunk_words(lo, hi) {
                let bits = c.bits[w] & mask & !mine.bits[w];
                n += bits.count_ones() as usize;
                mine.bits[w] |= bits;
            }
            mine.set += n;
            added += n;
        }
        self.len += added;
        added
    }

    /// Clears blocks `[from, to)`.
    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let (first, last) = (from >> CHUNK_SHIFT, (to - 1) >> CHUNK_SHIFT);
        let start = self.chunks.partition_point(|c| c.at < first);
        for c in self.chunks[start..].iter_mut().take_while(|c| c.at <= last) {
            if c.set == 0 {
                continue;
            }
            let (lo, hi) = Self::local(c.at, from, to);
            let mut gone = 0;
            for (w, mask) in chunk_words(lo, hi) {
                gone += (c.bits[w] & mask).count_ones() as usize;
                c.bits[w] &= !mask;
            }
            c.set -= gone;
            self.len -= gone;
        }
    }

    /// How many blocks of `[from, to)` are set.
    fn count_in(&self, from: usize, to: usize) -> usize {
        self.chunks_in(from, to)
            .iter()
            .filter(|c| c.set != 0)
            .map(|c| match Self::local(c.at, from, to) {
                (0, CHUNK) => c.set,
                (lo, hi) => chunk_words(lo, hi).map(|(w, mask)| (c.bits[w] & mask).count_ones() as usize).sum(),
            })
            .sum()
    }

    /// The set blocks of `[from, to)`, lowest first.
    fn iter_in(&self, from: usize, to: usize) -> impl Iterator<Item = usize> + '_ {
        self.chunks_in(from, to).iter().filter(|c| c.set != 0).flat_map(move |c| {
            let (lo, hi) = Self::local(c.at, from, to);
            let base = c.at << CHUNK_SHIFT;
            chunk_words(lo, hi).flat_map(move |(w, mask)| Ones(c.bits[w] & mask).map(move |o| base + w * 64 + o))
        })
    }

    /// Whether every block of `[from, to)` is set.
    fn all_in(&self, from: usize, to: usize) -> bool {
        from >= to || self.first_unset(from, to) == to
    }

    /// The first block of `[from, to)` not set; `to` if all are.
    fn first_unset(&self, from: usize, to: usize) -> usize {
        Self::first_in_none(self, None, from, to)
    }

    /// The first block of `[from, to)` set neither in `a` nor in `b`; `to`
    /// if there is none.
    fn first_in_none(a: &Blocks, b: Option<&Blocks>, from: usize, to: usize) -> usize {
        let mut at = from;
        while at < to {
            let n = at >> CHUNK_SHIFT;
            let (ca, cb) = (a.chunk(n), b.and_then(|b| b.chunk(n)));
            if ca.is_none() && cb.is_none() {
                return at;
            }
            if !ca.is_some_and(|c| c.set == CHUNK) && !cb.is_some_and(|c| c.set == CHUNK) {
                let (lo, hi) = Self::local(n, at, to);
                for (w, mask) in chunk_words(lo, hi) {
                    let free = !(ca.map_or(0, |c| c.bits[w]) | cb.map_or(0, |c| c.bits[w])) & mask;
                    if free != 0 {
                        return (n << CHUNK_SHIFT) + w * 64 + free.trailing_zeros() as usize;
                    }
                }
            }
            at = (n + 1) << CHUNK_SHIFT;
        }
        to
    }
}

#[cfg(test)]
impl FromIterator<usize> for Blocks {
    fn from_iter<I: IntoIterator<Item = usize>>(blocks: I) -> Blocks {
        let mut set = Blocks::new();
        for block in blocks {
            set.insert(block);
        }
        set
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them (always some
    /// of `protected`: what leaves that set leaves this one).
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Blocks,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: Blocks,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: Blocks,
    /// Runs of blocks Windows lately said are wholly uncommitted, by first
    /// block: the end block, whether the memory is a mapped view, and when
    /// it was said ([`hole_end`]). They do not overlap.
    holes: BTreeMap<usize, Hole>,
    /// Changes of `holes`, counted: what [`guard`]'s walk read of them
    /// ([`HolesAhead`]) stands while the count does.
    holes_changed: u64,
    /// Blocks [`protect`] is making read-only with the lock let go
    /// ([`guard_unlocked`]): until it notes them, nothing else changes their
    /// guards, opens them or decides about a write to them ([`wait_protecting`]).
    protecting: Blocks,
    /// With [`rest_now`]: when checks last made blocks read-only, and the
    /// blocks a write opened soon after, left open for a while.
    rests: Rests,
    /// With [`hole_bits`]: the blocks of `absent` inside a hole of `holes`
    /// that is fresh ([`fresh`]); and for each run of them noted, the
    /// millisecond of the answer that holds it, the earliest first
    /// ([`quiet_expire`]).
    quiet: Blocks,
    quiet_ends: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize, usize)>>,
}

impl Guards {
    const fn new() -> Guards {
        Guards {
            protected: Blocks::new(),
            clean: Blocks::new(),
            absent: Blocks::new(),
            aliased: Blocks::new(),
            viewed: Blocks::new(),
            holes: BTreeMap::new(),
            holes_changed: 0,
            protecting: Blocks::new(),
            rests: Rests::new(),
            quiet: Blocks::new(),
            quiet_ends: std::collections::BinaryHeap::new(),
        }
    }
}

/// How long a block that the title wrote soon after a check made it
/// read-only is left open, in microseconds (`VICTORY_GUARD_REST=1`;
/// `VICTORY_GUARD_REST_US=<n>` sets the time, 2,000 unless given; `alt`:
/// twenty seconds with, twenty without, in turn, each change said in the
/// log; off by default). 0 when off.
///
/// A block the title keeps writing is made read-only by a check, written
/// at once, opened by the fault, made read-only by the next check, and so
/// on: a `VirtualProtect` of ours and a fault of the title's each time. In
/// run BA's play the submitting thread spent 3.1 ms a frame in
/// `VirtualProtect` making blocks read-only again ([`protect`]) and 1.9 ms
/// waiting to hold the guards again after it, and the title's threads 6.6
/// ms a frame waiting in the fault handler for the guards' lock (360 guard
/// faults a frame in run BL). On the same title VIctory-7900XTX's fault
/// book found 220 of a frame's 586 faults within 3 ms of the check that had
/// made the block read-only, up to a dozen times a frame for one block
/// (its 7bc7c07). By hand on this machine (2026-11-26, a 1 GiB test
/// process): making one block read-only costs 2.3 us alone and 11.4 us with
/// 24 threads of the process running beside (the shootdown of their TLBs),
/// a copy of the block 3-4.5 us.
///
/// So a block opened by a write less than this long after a check made it
/// read-only rests: for this long after the write, and for at most
/// [`rest_checks`] checks of it, no check makes it read-only again, and
/// every check reports it written, as it reports any block that is not
/// guarded now (nothing written is missed: what is not read-only is never
/// clean). The first check after the rest makes it read-only again. Blocks
/// written once in a while are never left open: their next check guards
/// them as before, and copies nothing more. What it costs: a resting block
/// is copied at each check of it, written or not. A rest pays where the
/// title writes the block again between the checks of its rest (each such
/// write is a `VirtualProtect` of 11 us and a fault saved, the copy made
/// anyway); where it does not, the rest saves nothing and costs a copy
/// (3-4.5 us) at each of its checks, which the cap on checks bounds.
///
/// By hand (`bench_tests::checks_beside_a_writer_with_and_without_rests`:
/// 32 blocks of 256 written in turn, the range checked every 250 us, 16
/// threads running beside; agents' builds beside, so the times are rough):
/// with a write every 15 us (each block again every ~0.5 ms), without
/// rests every write faulted (13.8 us each) and the checks took 1.15 s of
/// 2 s; with rests of 4 ms and no cap on checks the writer wrote 1.4 times
/// as much, half its writes faulted (10.7 us) and the checks took 0.68 s,
/// but 3.5 times as many blocks were reported written (copies). With a
/// write every 120 us (each block again every ~4 ms) those rests saved no
/// `VirtualProtect` and reported 5 times as many blocks: so 2 ms and 2
/// checks by default.
fn rest_now() -> u64 {
    rest_in_turn(true)
}

/// [`rest_now`]; with `say`, a change of `alt`'s turn is said in the log
/// (not from the fault handler, which must not wait for the log's lock).
fn rest_in_turn(say: bool) -> u64 {
    #[cfg(test)]
    if REST_HERE.get() != 0 {
        return REST_HERE.get();
    }
    let on = match rest_mode() {
        0 => false,
        1 => true,
        _ => {
            let on = (crate::host::now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if say && LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "blocks written just after their check rest from now (VICTORY_GUARD_REST=alt): {}",
                    if on { "RESTING, left open a while" } else { "off, guarded again at the next check as usual" }
                );
            }
            on
        }
    };
    if on { rest_us() } else { 0 }
}

/// `VICTORY_GUARD_REST`: 0 off, 1 on, 2 in turn ([`rest_now`]).
fn rest_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GUARD_REST").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// `VICTORY_GUARD_REST_US`: a rest's length in microseconds ([`rest_now`]).
fn rest_us() -> u64 {
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *US.get_or_init(|| {
        std::env::var("VICTORY_GUARD_REST_US")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&us| us > 0)
            .unwrap_or(2000)
    })
}

/// `VICTORY_GUARD_REST_CHECKS`: the checks a rest lasts at most, 2 unless
/// given ([`rest_now`]): the next makes the block read-only again, however
/// much of the rest's time is left.
fn rest_checks() -> u32 {
    #[cfg(test)]
    if REST_CHECKS_HERE.get() != 0 {
        return REST_CHECKS_HERE.get();
    }
    static CHECKS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CHECKS.get_or_init(|| {
        std::env::var("VICTORY_GUARD_REST_CHECKS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2)
    })
}

#[cfg(test)]
std::thread_local! {
    /// Tests: the rest this thread's checks and faults use ([`rest_now`];
    /// 0: as the switch says), and the checks it lasts ([`rest_checks`]).
    static REST_HERE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static REST_CHECKS_HERE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Runs of more blocks than this made read-only at once are not noted in
/// [`Rests::armed`]: a range guarded for the first time (thousands of
/// blocks), not a block found written. One of those that the title keeps
/// writing rests from its next turn on.
const REST_RUN_MOST: usize = 64;

/// Entries [`Rests`] keeps before it lets the old ones go.
const RESTS_MOST: usize = 1 << 14;

/// The guards' memory of [`rest_now`], in microseconds of
/// `crate::host::now_us`.
struct Rests {
    /// Blocks resting now; their ends are in `until`. (A bit, so that a look
    /// at it costs what the guards' other looks do.)
    rested: Blocks,
    /// Of those, when each rest ends and the checks it has left.
    until: BTreeMap<usize, (u64, u32)>,
    /// Blocks a check made read-only lately, and when.
    armed: BTreeMap<usize, u64>,
}

impl Rests {
    const fn new() -> Rests {
        Rests { rested: Blocks::new(), until: BTreeMap::new(), armed: BTreeMap::new() }
    }

    /// A check made blocks `[from, to)` read-only at `now`.
    fn armed(&mut self, from: usize, to: usize, now: u64, rest: u64) {
        if to - from > REST_RUN_MOST {
            return;
        }
        if self.armed.len() >= RESTS_MOST {
            self.armed.retain(|_, at| now.saturating_sub(*at) < rest);
            if self.armed.len() >= RESTS_MOST {
                self.armed.clear();
            }
        }
        for block in from..to {
            self.armed.insert(block, now);
        }
    }

    /// A write opened `block` at `now`: whether it rests (a check made it
    /// read-only less than `rest` before), until `now + rest` or for
    /// `checks` checks.
    fn opened(&mut self, block: usize, now: u64, rest: u64, checks: u32) -> bool {
        let Some(at) = self.armed.remove(&block) else {
            return false;
        };
        if now.saturating_sub(at) >= rest {
            return false;
        }
        if self.until.len() >= RESTS_MOST {
            let ended: Vec<usize> = self.until.iter().filter(|&(_, &(end, _))| end <= now).map(|(&b, _)| b).collect();
            for b in ended {
                self.until.remove(&b);
                self.rested.remove(b);
            }
        }
        self.rested.insert(block);
        self.until.insert(block, (now + rest, checks));
        true
    }

    /// A check of `block` at `now`: whether it rests (and one check of its
    /// rest is spent); one whose rest is over is forgotten.
    fn rests(&mut self, block: usize, now: u64) -> bool {
        if !self.rested.contains(block) {
            return false;
        }
        if let Some((end, left)) = self.until.get_mut(&block)
            && *end > now
            && *left > 0
        {
            *left -= 1;
            return true;
        }
        self.rested.remove(block);
        self.until.remove(&block);
        false
    }

    /// Blocks `[from, to)` show something new (the runtime mapped or
    /// unmapped them): none of them rests, and no check made them read-only
    /// lately. (A look into each map, mostly finding nothing: the runtime
    /// remaps thousands of times a frame in cutscenes.)
    fn remapped(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        self.rested.remove_range(from, to);
        let armed: Vec<usize> = self.armed.range(from..to).map(|(&b, _)| b).collect();
        for b in armed {
            self.armed.remove(&b);
        }
        let until: Vec<usize> = self.until.range(from..to).map(|(&b, _)| b).collect();
        for b in until {
            self.until.remove(&b);
        }
    }
}

/// Since the last frame line ([`rest_line`]): blocks checks made read-only
/// and their `VirtualProtect` calls; blocks a write opened soon after their
/// check, left to rest; blocks checks found resting and left open.
static RESTED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// With the frame line, while `VICTORY_GUARD_REST` is on or in turn: what
/// the checks made read-only and what rested, per frame over `frames`.
pub(crate) fn rest_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [blocks, calls, began, left] = [0, 1, 2, 3].map(|i| RESTED[i].swap(0, Relaxed) as f64 / frames);
    if rest_mode() == 0 {
        return;
    }
    tracing::info!(
        "guards per frame: {blocks:.0} blocks made read-only by checks in {calls:.0} calls; {began:.0} opened by a write \
         soon after their check, left to rest; {left:.0} checks of a resting block left it open (VICTORY_GUARD_REST, {} us)",
        rest_us()
    );
}

/// What [`Guards::holes`] keeps of one answer of Windows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hole {
    end: usize,
    mapped: bool,
    /// The frame (`crate::frame::current`) and the millisecond
    /// (`crate::host::now_ms`) of the answer.
    frame: u64,
    ms: u64,
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
///
/// `VICTORY_VIEW_KNOWN=alt`: twenty seconds with, twenty without, in turn,
/// each change said in the log, for measuring it in one scene. (Blocks
/// noted in one phase are forgotten at a remapping in either.)
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
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
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Behind a lock that askings which change nothing share ([`writes_shared`])
/// and plain looks at the sets ([`guards_shared`]); everything that changes
/// the guards holds it alone ([`guards`]), as when it was a mutex.
static GUARDS: RwLock<Guards> = RwLock::new(Guards::new());

fn guards() -> std::sync::RwLockWriteGuard<'static, Guards> {
    GUARDS.write().unwrap_or_else(|e| e.into_inner())
}

/// The guards, for a look: held with other lookers, never across a change.
fn guards_shared() -> std::sync::RwLockReadGuard<'static, Guards> {
    GUARDS.read().unwrap_or_else(|e| e.into_inner())
}

/// The guards held alone ([`guards`]), let go for a while by
/// [`Held::without`].
struct Held(Option<std::sync::RwLockWriteGuard<'static, Guards>>);

impl Held {
    fn new() -> Held {
        Held(Some(guards()))
    }

    /// Runs `f` with the guards let go if `unlock` (else held), and holds
    /// them again.
    fn without<T>(&mut self, unlock: bool, f: impl FnOnce() -> T) -> T {
        if !unlock {
            return f();
        }
        self.0 = None;
        let done = f();
        self.0 = Some(guards());
        done
    }
}

impl std::ops::Deref for Held {
    type Target = Guards;

    fn deref(&self) -> &Guards {
        self.0.as_deref().expect("the guards are held")
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.0.as_deref_mut().expect("the guards are held")
    }
}

/// Whether [`guard`] lets the guards' lock go while Windows makes the blocks
/// it guards again read-only (`VICTORY_GUARD_UNLOCKED`; on unless `0`).
///
/// The submitting thread holds the lock through each whole check, its
/// system calls inside: run AM's frame has 3.4 ms of `VirtualProtect` and
/// 1.2 ms of `VirtualQuery` in [`guard`], and the title's threads waited
/// 6.0 ms a frame for the lock in the fault handler, the relay 1.2 ms in
/// [`writing`]. With the switch the lock is let go around the
/// `VirtualProtect` calls of [`protect`] (the questions of `hole_end`
/// stay under it). By hand (`bench_tests::faults_beside_a_check_that_guards_again`,
/// a check guarding 256 runs again beside a thread whose writes fault): the
/// longest wait of a write 400-560 us held, 113-180 us let go, and 1.7
/// times the writes done meanwhile. Runs AZ and BA (2026-11-26): pictures
/// right, no fault lost; with the rest of that branch the submitting thread
/// 52 ms a frame against 60, the frame 93 ms against 96-100.
///
/// Why the order stays right. Today the lock makes "make the run read-only,
/// then note it guarded and clean" one step for every other holder. Let go,
/// the run's blocks (and the other views [`protect`] will make read-only)
/// are first put in [`Guards::protecting`], with the lock held, and taken
/// out with the lock held, in the same step that notes what Windows did
/// for them, after the last system call on them. Meanwhile:
/// - every step that changes the guards of such a block, opens it or
///   decides about a write to it waits until it is out ([`wait_protecting`],
///   the lock let go while waiting): [`open`] (so the fault handler,
///   [`writing`] and [`title_memory_written`], and the other views `open`
///   opens with a block, read once with the lock held), `views_mapped`,
///   [`title_memory_remapped`]. So each of those sees the block either
///   before the check took it (it was neither guarded nor clean: as if the
///   step had come before the check) or after it was noted (as if after);
///   never between. A write that faults on a block of the run, read-only
///   already and not yet noted, waits in the handler until it is noted,
///   then opens it and counts it written, as it would have waited for the
///   lock; it is never left to `writable` (which would let it crash).
/// - the steps that do not wait touch only other blocks, or only look:
///   a look (`known_view`, [`block_clean`]) finds the block not yet
///   guarded, which only sends its caller the longer way (the kernel, a
///   check that waits for the journal the check holds);
///   [`title_memory_committed`] changes the holes and blocks noted absent,
///   and the run's blocks are not absent (the walk took them out), while
///   the questions of `hole_end` about blocks Windows would not make
///   read-only are asked after, with the lock held.
/// - the walk itself looks at the guards again once it holds the lock
///   (blocks opened or remapped meanwhile are found as they are now; the
///   holes it read are read again when their count of changes moved).
///
/// No wait can close a circle: a waiter lets the guards' lock go and holds
/// none of the locks the check takes (the journal is the check's own; the
/// handler and the runtime's steps hold neither it nor those of
/// `crate::inplace` when they wait), and the check takes no lock between
/// letting the guards go and taking them again: it only asks Windows.
fn guard_unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_UNLOCKED").is_none_or(|v| v != "0"));
    #[cfg(test)]
    let on = on || UNLOCKED_HERE.get();
    on
}

#[cfg(test)]
std::thread_local! {
    /// Tests: this thread's checks let the lock go ([`guard_unlocked`]).
    static UNLOCKED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Blocks [`protect`] took out of [`Guards::protecting`], counted (under the
/// guards' lock), and woken at each: what waits for them waits for the
/// count to move.
static PROTECTED: Mutex<u64> = Mutex::new(0);
static PROTECTED_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// With the guards held: waits, letting them go, while `busy` says a block
/// it is about is being made read-only ([`Guards::protecting`]), and holds
/// them again; `busy` is asked again each time.
fn wait_protecting(g: &mut Held, busy: impl Fn(&Guards) -> bool) {
    while !g.protecting.is_empty() && busy(g) {
        protecting_moved(g);
    }
}

/// With the guards held and blocks in [`Guards::protecting`]: lets the
/// guards go until [`protect`] takes some out (read under them: no wake is
/// missed), and holds them again.
fn protecting_moved(g: &mut Held) {
    let seen = *PROTECTED.lock().unwrap_or_else(|e| e.into_inner());
    g.without(true, || {
        let mut now = PROTECTED.lock().unwrap_or_else(|e| e.into_inner());
        while *now == seen {
            now = PROTECTED_MOVED.wait(now).unwrap_or_else(|e| e.into_inner());
        }
    });
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    guards_shared().clean.contains(page)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    // Looked at under the lock the walk holds: one taking of it, not two
    // (run AM: 0.2 ms a frame waiting for the shared one).
    let mut g = Held::new();
    if (ask || !known_view_in(&g, start)) && !g.without(true, || is_view(start, end)) {
        return None;
    }
    install();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // A frame asks about some 10 GB of view memory, 150,000 blocks, nearly
    // all clean: looked up one block at a time they were most of the 60-90
    // ms a frame of "guards" (run 54); read off a `BTreeSet` into a vector
    // at each check, and walked block by block, still 4.3 ms a frame (run
    // AM). The clean blocks are bits now, read 64 to a word, with nothing
    // gathered (`bench_tests`: 2,000 checks of 75 clean blocks 1.6-2.1 ms,
    // now 0.03-0.06 ms). What the walk adds to them meanwhile is the block
    // it stands at or ones before it.
    if g.clean.all_in(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    // Absent blocks inside holes Windows lately said are there pass as
    // clean ones do ([`hole_bits`]), once the answers grown old are let go.
    if !g.quiet_ends.is_empty() {
        quiet_expire(&mut g, crate::host::now_ms());
    }
    let quiet = !g.quiet.is_empty() && hole_bits();
    if quiet && Blocks::first_in_none(&g.clean, Some(&g.quiet), first, last) == last {
        HOLE_BITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Some(written);
    }
    let mut uncommitted_until = first;
    let mut holes = HolesAhead::new(last);
    // Blocks resting ([`rest_now`]) are reported and left open; the clock
    // is read once, at the first one (a rest is milliseconds long).
    let rest = rest_now();
    let mut now = None;
    let mut at = first;
    loop {
        // Blocks `[at, block)` are clean, or absent inside the hole the
        // walk last found (or, with `quiet`, inside any hole lately said):
        // nothing to do there but end the run before them.
        let block = if quiet { next_open_quiet(&g, at, uncommitted_until, last) } else { next_open(&g, at, uncommitted_until, last) };
        if block > at {
            end_run(&mut g, &mut run, at, &mut holes);
        }
        if block == last {
            break;
        }
        let mut open = true;
        if g.absent.contains(block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region
            // (and a recent asking's: [`hole_end`]).
            if block >= uncommitted_until {
                uncommitted_until = hole_end_ahead(&mut g, block, &mut holes);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(block);
                g.quiet.remove(block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
                end_run(&mut g, &mut run, block, &mut holes);
            } else if rest > 0
                && !g.rests.rested.is_empty()
                && g.rests.rests(block, *now.get_or_insert_with(crate::host::now_us))
            {
                // The title wrote it just after a check guarded it: left open,
                // reported written (above) until its rest is over.
                RESTED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                end_run(&mut g, &mut run, block, &mut holes);
            } else {
                run.get_or_insert(block);
            }
        } else {
            end_run(&mut g, &mut run, block, &mut holes);
        }
        at = block + 1;
    }
    end_run(&mut g, &mut run, last, &mut holes);
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
///
/// Holes Windows lately said are wholly uncommitted are passed over
/// without asking again ([`known_hole`]), and the holes it says are kept.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // Held from the look at what is kept to the noting of the answer: a
        // mapping told meanwhile (`title_memory_remapped`) comes after the
        // note and forgets it.
        let mut g = guards();
        let known = if holes { known_hole(&g, at / BLOCK) } else { None };
        if let Some((hole_end, mapped)) = known {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::HolesKnown, 1);
            if mapped {
                return true;
            }
            let next = hole_end.saturating_mul(BLOCK);
            if next <= at || next >= end {
                return true;
            }
            at = next;
            continue;
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        // A copy of a physical page kept at a page-mapping copy's target
        // counts as view memory (`crate::alias::is_copy`).
        if info.Type == MEM_MAPPED || crate::alias::is_copy(at) {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HolesAsked, 1);
        let (first, last) = ((info.BaseAddress as usize).div_ceil(BLOCK), next / BLOCK);
        if first < last {
            note_hole(&mut g, first, last, false);
        }
        drop(g);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
///
/// In three steps: what to do, with the guards held (the other views, the
/// blocks put in [`Guards::protecting`]); the system calls, with the guards
/// let go if [`guard_unlocked`]; what Windows did noted, and the blocks
/// taken out of `protecting`, with the guards held again.
fn protect(g: &mut Held, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    // The other views, as each block's turn finds them: (view, guarded
    // before the check) in order, a view once per block that shows it.
    let mut others: Vec<(usize, bool)> = Vec::new();
    if view_guard() {
        for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
            let first = (base / BLOCK).max(from);
            let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
            for block in first..last {
                let shown = crate::inplace::pages_of(block * BLOCK);
                // One view (nothing else to guard), or the page the title shows
                // wherever it has nothing (thousands of views): each view of
                // that one is guarded by itself.
                if shown.len() < 2 || shown.len() > MANY_VIEWS {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for alias in shown {
                    let alias = alias / BLOCK;
                    g.aliased.insert(alias);
                    if alias != block {
                        others.push((alias, g.protected.contains(alias)));
                    }
                }
            }
        }
    }
    g.protecting.insert_range(from, to);
    for &(alias, guarded) in &others {
        if !guarded {
            g.protecting.insert(alias);
        }
    }
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) } != 0;
        // The error of one that failed, for the diagnostics of `protect_noted`.
        (!done).then(|| std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
    };
    let (whole, each, others_done) = g.without(guard_unlocked(), || {
        let whole = set(from, to).is_none();
        let each: Vec<Option<i32>> = if whole { Vec::new() } else { (from..to).map(|block| set(block, block + 1)).collect() };
        // As the other views were taken one by one after the run: a view
        // guarded already (before the check, by the run, or just now) is
        // left.
        let mut done: Vec<usize> = Vec::new();
        for &(alias, guarded) in &others {
            let by_run = (from..to).contains(&alias) && (whole || each[alias - from].is_none());
            if guarded || by_run || done.contains(&alias) {
                continue;
            }
            if set(alias, alias + 1).is_none() {
                done.push(alias);
            }
        }
        (whole, each, done)
    });
    protect_noted(g, from, to, whole, &each);
    for alias in others_done {
        protect_alias_noted(g, alias);
    }
    {
        use std::sync::atomic::Ordering::Relaxed;
        RESTED[0].fetch_add((to - from) as u64, Relaxed);
        RESTED[1].fetch_add(if whole { 1 } else { 1 + (to - from) as u64 }, Relaxed);
    }
    // When, for a write that opens them soon after ([`Rests::opened`]).
    let rest = rest_now();
    if rest > 0 {
        g.rests.armed(from, to, crate::host::now_us(), rest);
    }
    g.protecting.remove_range(from, to);
    for &(alias, _) in &others {
        g.protecting.remove(alias);
    }
    *PROTECTED.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    PROTECTED_MOVED.notify_all();
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
pub(crate) fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Notes `block` read-only ([`protect`] made it so): another view of pages
/// a copy has just taken through its own view, so that a write through this
/// one faults too. It is not noted clean. The copies of title memory at this
/// view have not taken the pages: noted clean, the block would not be
/// reported written at their next check and they would keep the bytes from
/// before the write (`guard` reports a block that is read-only and not clean
/// as written, then notes it clean). A block that is clean stays so: had the
/// pages been written since its copies took them, `open` would have opened
/// it.
fn protect_alias_noted(g: &mut Guards, block: usize) {
    g.protected.insert(block);
    if view_known() {
        g.viewed.insert(block);
    }
}

/// What [`protect`] found making `[from, to)` read-only, noted: all of it
/// (`whole`), or block by block as `each` says (`None`: done; else the
/// error), the blocks Windows would not make read-only noted absent where
/// they are not committed.
fn protect_noted(g: &mut Guards, from: usize, to: usize, whole: bool, each: &[Option<i32>]) {
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert_range(from, to);
        g.clean.insert_range(from, to);
        if view_known() {
            g.viewed.insert_range(from, to);
        }
    };
    if whole {
        note(g, from, to);
        return;
    }
    for (block, failed) in (from..to).zip(each) {
        let Some(_error) = failed else {
            note(g, block, block + 1);
            continue;
        };
        if block < hole_end(g, block) {
            g.absent.insert(block);
        } else {
            // Diagnostics: a committed block that will not be made
            // read-only is reported written at every asking and never
            // known to the guards (one in 1,024 said).
            #[cfg(feature = "diagnostics")]
            {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(1024) {
                    tracing::info!(
                        "a block of title memory could not be made read-only (one in 1024): {:#x}: {}",
                        block * BLOCK,
                        std::io::Error::from_raw_os_error(*_error)
                    );
                }
            }
        }
    }
    // The blocks just noted absent lie in holes just said or known fresh.
    if hole_bits_mode() != 0 {
        quiet_mark_fresh(g, from, to, crate::host::now_ms());
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`,
/// and whether they are a mapped view (of a section whose pages are not
/// committed); `(block, false)` if `block` is committed.
fn uncommitted_end(block: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (
            ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).max(block),
            info.Type == MEM_MAPPED,
        )
    } else {
        (block, false)
    }
}

/// How long Windows' word that blocks are wholly uncommitted is taken
/// without asking again, in milliseconds (`VICTORY_ABSENT_KEEP`; 1,000 by
/// default, `0` asks at every check as before), across frames.
///
/// (100 ms and never past the frame it was said in, first: run AB's play
/// still asked about 4,700 runs a frame, each once a frame, 8.3 ms a frame
/// of `VirtualQuery`, for holes the runtime's remappings, which forget them
/// at once, change a few times a frame at most.)
///
/// A copy of view memory whose range holds blocks that are not mapped
/// (placeholders between the title's views, free memory beside them) asked
/// Windows about every such run of blocks at every check: they are never
/// clean, so no check of the range was answered from the guards alone. In
/// run P that was 10.7 ms a frame of the submitting thread in
/// `VirtualQuery` under `guard`, some 2,000 checks a frame. What maps such
/// blocks is the runtime's (`xmem`: [`title_memory_remapped`], and
/// [`title_memory_committed`] for its commits into the title's
/// reservations), and its word forgets them at once. A commit nobody tells
/// (the title's own `VirtualAlloc` into a reservation of its own, memory
/// beside a view that another allocation takes) is seen at the first check
/// after the answer is this old, or in the next frame: until then the
/// copies there keep the zeros they took.
fn hole_keep() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_ABSENT_KEEP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1000)
    })
}

/// The most runs of blocks [`Guards::holes`] keeps (the title's views lie
/// in some thousands of placeholders); past it, all are forgotten.
const HOLES_MOST: usize = 1 << 16;

/// Whether `hole` may still be taken at frame `now_frame` and millisecond
/// `now_ms`, kept `keep` ms.
fn fresh(hole: &Hole, now_frame: u64, now_ms: u64, keep: u64) -> bool {
    let _ = (hole.frame, now_frame);
    keep > 0 && now_ms.saturating_sub(hole.ms) < keep
}

/// The run of blocks from `block` on that [`Guards::holes`] holds, if what
/// it says is fresh: (end block, whether a mapped view).
fn known_hole(g: &Guards, block: usize) -> Option<(usize, bool)> {
    let keep = hole_keep();
    if keep == 0 || g.holes.is_empty() {
        return None;
    }
    known_hole_at(g, block, crate::frame::current(), crate::host::now_ms(), keep)
}

/// [`known_hole`] at frame `frame` and millisecond `ms`, kept `keep` ms.
fn known_hole_at(g: &Guards, block: usize, frame: u64, ms: u64, keep: u64) -> Option<(usize, bool)> {
    let (_, hole) = g.holes.range(..=block).next_back()?;
    (hole.end > block && fresh(hole, frame, ms, keep)).then_some((hole.end, hole.mapped))
}

/// [`uncommitted_end`]'s end, from [`Guards::holes`] while it is fresh
/// ([`hole_keep`]), else asked of Windows and kept there.
fn hole_end(g: &mut Guards, block: usize) -> usize {
    if let Some((end, _)) = known_hole(g, block) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HolesKnown, 1);
        return end;
    }
    hole_end_asked(g, block)
}

/// [`hole_end`] where [`Guards::holes`] says nothing fresh: Windows is
/// asked, and its answer kept.
fn hole_end_asked(g: &mut Guards, block: usize) -> usize {
    let (end, mapped) = uncommitted_end(block);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::HolesAsked, 1);
    note_hole(g, block, end, mapped);
    end
}

/// What [`guard`]'s walk keeps of [`Guards::holes`] between its lookups
/// ([`hole_end_ahead`]): the hole at or before the block it looks up and
/// the next [`HOLES_AHEAD`] after it in the range, read off the map in one
/// pass and standing until the map changes (`Guards::holes_changed`); and
/// the clock, read once until the walk next asks Windows anything. Run AM's
/// frame passed over some 10,500 absent blocks on a recent answer: 1.0 ms
/// a frame searching the map, 0.3 ms reading the clock (34 ns a reading).
struct HolesAhead {
    /// Holes by first block, as the map has them.
    holes: Vec<(usize, Hole)>,
    /// Of those, the last that begins at or before the block last looked up.
    at: usize,
    /// Whether `holes` ends where the map does (before `last`).
    whole: bool,
    /// The map's count of changes when it was read; `None` before.
    read: Option<u64>,
    /// The end of the walk's range: no block there or past it is looked up.
    last: usize,
    /// The frame and millisecond whose answers are fresh ([`fresh`]), read
    /// at the first look since the walk began or last asked Windows (a
    /// guard or a question of its own): between those the walk takes some
    /// microseconds, not the milliseconds an answer is kept.
    now: Option<(u64, u64)>,
}

/// Holes [`HolesAhead`] reads at once past the one at the block looked up.
const HOLES_AHEAD: usize = 32;

impl HolesAhead {
    fn new(last: usize) -> HolesAhead {
        HolesAhead { holes: Vec::new(), at: 0, whole: false, read: None, last, now: None }
    }

    /// The walk asked Windows something, which takes time: the clock is read
    /// again at the next look.
    fn asked(&mut self) {
        self.now = None;
    }

    /// [`known_hole`] of `block`, as the map says now. Blocks are looked up
    /// in order: none lower than the one before.
    fn known(&mut self, g: &Guards, block: usize) -> Option<(usize, bool)> {
        let keep = hole_keep();
        if keep == 0 || g.holes.is_empty() {
            return None;
        }
        if self.read == Some(g.holes_changed) {
            while self.holes.get(self.at + 1).is_some_and(|&(start, _)| start <= block) {
                self.at += 1;
            }
            // Holes past those read may begin at or before `block`.
            if self.at + 1 == self.holes.len() && !self.whole {
                self.read = None;
            }
        }
        if self.read != Some(g.holes_changed) {
            self.holes.clear();
            self.holes.extend(g.holes.range(..=block).next_back().map(|(&start, &hole)| (start, hole)));
            let before = self.holes.len();
            let ahead = g.holes.range(block + 1..self.last.max(block + 1)).take(HOLES_AHEAD);
            self.holes.extend(ahead.map(|(&start, &hole)| (start, hole)));
            self.whole = self.holes.len() - before < HOLES_AHEAD;
            self.at = 0;
            self.read = Some(g.holes_changed);
        }
        let &(start, hole) = self.holes.get(self.at)?;
        if start > block || hole.end <= block {
            return None;
        }
        let (frame, ms) = *self.now.get_or_insert_with(|| (crate::frame::current(), crate::host::now_ms()));
        fresh(&hole, frame, ms, keep).then_some((hole.end, hole.mapped))
    }
}

/// [`hole_end`] for [`guard`]'s walk, looking in what it read of the map.
/// A fresh hole found there notes its absent blocks quiet ([`hole_bits`]).
fn hole_end_ahead(g: &mut Guards, block: usize, ahead: &mut HolesAhead) -> usize {
    if let Some((end, _)) = ahead.known(g, block) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HolesKnown, 1);
        if let Some(&(_, hole)) = ahead.holes.get(ahead.at) {
            quiet_mark(g, block, end, hole.ms);
        }
        return end;
    }
    ahead.asked();
    hole_end_asked(g, block)
}

/// Whether [`guard`]'s walk passes over the absent blocks inside a hole
/// Windows lately said is there as it passes over clean ones, by bits
/// (`VICTORY_HOLE_BITS=1`; `alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log; off by default).
///
/// Ranges of view memory that hold unmapped blocks (placeholders between
/// the title's views) are never all clean, so every check of them walked
/// them, and every absent block past the hole it last found was looked up
/// in [`Guards::holes`]: run BL's play 8,823 such lookups a frame against
/// 510 questions to Windows; run BA's profile 1.7 ms a frame of the
/// submitting thread in [`HolesAhead::known`] (most of it reading the map
/// again: each answer of Windows changes it) beside 1.1 ms of
/// `VirtualQuery`, all with the guards held. With the switch the absent
/// blocks inside such a hole are bits ([`Guards::quiet`]), set when the
/// hole is said or first found, and gone when it is forgotten, when the
/// block is no longer absent, and when the answer is older than
/// [`hole_keep`] (the same freshness the lookups use). A check whose blocks
/// are all clean or quiet then returns at once, and a walk passes over
/// quiet blocks 64 to a word; what it does at the others is unchanged.
/// (Windows is asked as often as before: when no fresh answer holds.)
fn hole_bits() -> bool {
    match hole_bits_mode() {
        0 => false,
        1 => true,
        _ => {
            let on = (crate::host::now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "absent blocks inside known holes are passed over by bits from now (VICTORY_HOLE_BITS=alt): {}",
                    if on { "BITS" } else { "off, looked up one by one as usual" }
                );
            }
            on
        }
    }
}

/// `VICTORY_HOLE_BITS`: 0 off, 1 on, 2 in turn ([`hole_bits`]). The bits
/// are kept whenever it is not 0, so that a turn begins with them right.
fn hole_bits_mode() -> u8 {
    #[cfg(test)]
    if HOLE_BITS_HERE.get() {
        return 1;
    }
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_HOLE_BITS").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

#[cfg(test)]
std::thread_local! {
    /// Tests: this thread's checks keep and use the bits ([`hole_bits`]).
    static HOLE_BITS_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs noted in [`Guards::quiet_ends`] past which all bits are let go (a
/// walk then looks the holes up again and notes them anew).
const QUIET_MOST: usize = 1 << 16;

/// Since the last frame line ([`hole_bits_line`]): checks answered by the
/// clean and quiet bits alone, and absent blocks noted quiet.
static HOLE_BITS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// With the frame line, while `VICTORY_HOLE_BITS` is on or in turn.
pub(crate) fn hole_bits_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [checks, marked] = [0, 1].map(|i| HOLE_BITS[i].swap(0, Relaxed) as f64 / frames);
    if hole_bits_mode() == 0 {
        return;
    }
    tracing::info!(
        "guards per frame: {checks:.0} checks answered by the bits of clean blocks and of absent ones inside known holes alone, \
         {marked:.0} absent blocks noted inside known holes (VICTORY_HOLE_BITS)"
    );
}

/// The absent blocks of `[from, to)` lie inside a hole Windows said at
/// millisecond `ms`: noted quiet ([`hole_bits`]) until that answer is no
/// longer fresh ([`quiet_expire`]).
fn quiet_mark(g: &mut Guards, from: usize, to: usize, ms: u64) {
    if hole_bits_mode() == 0 || from >= to || g.absent.is_empty() {
        return;
    }
    if g.quiet_ends.len() >= QUIET_MOST {
        quiet_clear(g);
    }
    let added = g.quiet.insert_and(&g.absent, from, to);
    if added > 0 {
        HOLE_BITS[1].fetch_add(added as u64, std::sync::atomic::Ordering::Relaxed);
        g.quiet_ends.push(std::cmp::Reverse((ms, from, to)));
    }
}

/// No block is quiet any more.
fn quiet_clear(g: &mut Guards) {
    g.quiet = Blocks::new();
    g.quiet_ends.clear();
}

/// At millisecond `now`: the quiet blocks of answers no longer fresh
/// ([`fresh`]) are quiet no longer, but where a fresher hole holds them.
fn quiet_expire(g: &mut Guards, now: u64) {
    let keep = hole_keep();
    while let Some(&std::cmp::Reverse((ms, from, to))) = g.quiet_ends.peek() {
        if keep > 0 && now.saturating_sub(ms) < keep {
            break;
        }
        g.quiet_ends.pop();
        g.quiet.remove_range(from, to);
        // What holes said again since hold.
        quiet_mark_fresh(g, from, to, now);
    }
}

/// Notes quiet the absent blocks of `[from, to)` that lie inside a hole
/// fresh at millisecond `now` (each with its hole's answer).
fn quiet_mark_fresh(g: &mut Guards, from: usize, to: usize, now: u64) {
    let keep = hole_keep();
    if hole_bits_mode() == 0 || keep == 0 || from >= to || g.holes.is_empty() {
        return;
    }
    let before = g.holes.range(..from).next_back().filter(|(_, h)| h.end > from);
    let fresher: Vec<(usize, usize, u64)> = before
        .into_iter()
        .chain(g.holes.range(from..to))
        .filter(|(_, h)| fresh(h, 0, now, keep))
        .map(|(&s, h)| (s.max(from), h.end.min(to), h.ms))
        .collect();
    for (a, b, ms) in fresher {
        quiet_mark(g, a, b, ms);
    }
}

/// Ends the walk's run of blocks to guard again at `to` ([`protect`]).
fn end_run(g: &mut Held, run: &mut Option<usize>, to: usize, holes: &mut HolesAhead) {
    if let Some(from) = run.take() {
        protect(g, from, to);
        holes.asked();
    }
}

/// Windows said blocks `[block, end)` are wholly uncommitted (nothing if
/// `end` is `block`: that block is committed). What the holes held of them
/// is older and goes.
fn note_hole(g: &mut Guards, block: usize, end: usize, mapped: bool) {
    let keep = hole_keep();
    if keep == 0 {
        return;
    }
    g.holes_changed += 1;
    forget_holes(g, block, end.max(block + 1));
    if end > block {
        if g.holes.len() >= HOLES_MOST {
            g.holes.clear();
            quiet_clear(g);
        }
        let hole = Hole { end, mapped, frame: crate::frame::current(), ms: crate::host::now_ms() };
        g.holes.insert(block, hole);
        quiet_mark(g, block, end, hole.ms);
    }
}

/// Blocks `[from, to)` may have been committed: [`Guards::holes`] keeps
/// only what it said of the blocks beside them (and none of these is quiet,
/// [`hole_bits`]).
fn forget_holes(g: &mut Guards, from: usize, to: usize) {
    if !g.quiet.is_empty() {
        g.quiet.remove_range(from, to);
    }
    if g.holes.is_empty() || from >= to {
        return;
    }
    g.holes_changed += 1;
    let before = g.holes.range(..from).next_back().filter(|(_, h)| h.end > from).map(|(&s, _)| s);
    let met: Vec<usize> = before.into_iter().chain(g.holes.range(from..to).map(|(&s, _)| s)).collect();
    for start in met {
        let Some(hole) = g.holes.remove(&start) else {
            continue;
        };
        if start < from {
            g.holes.insert(start, Hole { end: from, ..hole });
        }
        if hole.end > to {
            g.holes.insert(to, hole);
        }
    }
}

/// The runtime committed title memory `[at, at + len)` in a reservation
/// (`xmem`'s `XMemVirtualAlloc` with an address): blocks there that were
/// noted absent are looked at again at their next check, and what
/// [`hole_end`] kept of them goes.
pub fn title_memory_committed(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut g = guards();
    let g = &mut *g;
    let (from, to) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    forget_holes(g, from, to);
    g.absent.remove_range(from, to);
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
///
/// It waits while a block it would open, here or at another view, is being
/// made read-only ([`Guards::protecting`]): the other views are read once,
/// with the guards held, after the wait.
fn open(g: &mut Held, start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mut views: Vec<(usize, Vec<usize>)> = Vec::new();
    loop {
        views.clear();
        if start < end && !g.aliased.is_empty() {
            views.extend(g.aliased.iter_in(first, last).map(|block| (block, crate::inplace::pages_of(block * BLOCK))));
        }
        let busy = |g: &Guards| {
            g.protecting.count_in(first, last) != 0
                || views.iter().any(|(_, shown)| shown.iter().any(|&alias| g.protecting.contains(alias / BLOCK)))
        };
        if g.protecting.is_empty() || !busy(g) {
            break;
        }
        protecting_moved(g);
    }
    let any = open_here(g, start, end);
    for (block, shown) in views {
        g.aliased.remove(block);
        for alias in shown {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(alias / BLOCK);
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g.protected.iter_in(start / BLOCK, end.div_ceil(BLOCK)).collect();
    // A block a check made read-only just before rests ([`rest_now`]).
    // (Called from the fault handler too: nothing said in the log.)
    let rest = if blocks.is_empty() || g.rests.armed.is_empty() { 0 } else { rest_in_turn(false) };
    let now = if rest > 0 { crate::host::now_us() } else { 0 };
    for &block in &blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        // Copies of the page there are taken again (`crate::alias`).
        crate::alias::source_opened(block * BLOCK, BLOCK);
        if rest > 0 && g.rests.opened(block, now, rest, rest_checks()) {
            RESTED[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut Held::new(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = Held::new();
        let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
        wait_protecting(&mut g, |g| g.protecting.count_in(first, last) != 0);
        let guarded = g.protected.count_in(first, last) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = Held::new();
    wait_protecting(&mut g, |g| {
        ranges.iter().any(|&(at, n)| g.protecting.count_in(at / BLOCK, at.saturating_add(n).div_ceil(BLOCK)) != 0)
    });
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // The pages may have been committed through the new view (a section
        // that is only reserved): they are no hole at the others either.
        forget_holes(&mut g, at / BLOCK, at.saturating_add(n).div_ceil(BLOCK));
    }
}

/// Who holds which title memory: reads in place by range, shared, and the
/// runtime's changes of what title memory shows by range, alone. A change
/// waits only for reads of memory it touches, and a read only for changes
/// of the memory it reads (one lock for the whole address space was held
/// by the runtime's 1,459 remappings a world frame for 778 ms of it, and
/// texture refreshes stood behind it for 490-1,040 ms a frame, whatever
/// they read: MOONSHOT.md, run 6).
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::Mutex<Holders> = std::sync::Mutex::new(Holders { reading: Vec::new(), remapping: Vec::new(), waiting: Vec::new() });
/// Woken whenever a holder lets go.
static LET_GO: std::sync::Condvar = std::sync::Condvar::new();

/// Ranges `[from, to)` held, one entry per holder.
struct Holders {
    reading: Vec<(usize, usize)>,
    remapping: Vec<(usize, usize)>,
    /// Changes waiting for reads to end: reads of their memory that begin
    /// meanwhile wait behind them, so that a change is not put off for
    /// ever by reads that keep coming.
    waiting: Vec<(usize, usize)>,
}

fn held() -> std::sync::MutexGuard<'static, Holders> {
    MAPPING.lock().unwrap_or_else(|e| e.into_inner())
}

/// `[at, at + len)` as a range of at least one byte.
fn range(at: usize, len: usize) -> (usize, usize) {
    (at, at.saturating_add(len.max(1)))
}

fn meets(held: &[(usize, usize)], (from, to): (usize, usize)) -> bool {
    held.iter().any(|&(a, b)| a < to && from < b)
}

fn let_go(list: &mut Vec<(usize, usize)>, range: (usize, usize)) {
    if let Some(i) = list.iter().position(|&r| r == range) {
        list.swap_remove(i);
    }
}

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    range: (usize, usize),
    /// When the lock was asked for (the frame log) and had (diagnostics:
    /// [`REMAPS`]).
    asked: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    had: std::time::Instant,
}

/// Since the last frame-rate line (diagnostics): the runtime's remappings,
/// the microseconds it waited for the lock and those it held it, and of
/// those the microseconds in [`title_memory_remapped`] and in
/// `crate::inplace::title_memory_view`.
#[cfg(feature = "diagnostics")]
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Adds its time to [`REMAPS`]' place `.0` when dropped.
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) usize, pub(crate) std::time::Instant);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        REMAPS[self.0].fetch_add(self.1.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.asked.elapsed().as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        REMAPS[2].fetch_add(self.had.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        let_go(&mut held().remapping, self.range);
        LET_GO.notify_all();
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory `[at, at + len)`: until what it returns is dropped, no read in
/// place of memory there is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the
/// unmapping and the mapping again, for all the memory the change touches
/// (an unmapping takes whole views away and maps what stays again).
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    let asked = std::time::Instant::now();
    let range = range(at, len);
    let mut h = held();
    if meets(&h.reading, range) {
        h.waiting.push(range);
        while meets(&h.reading, range) {
            h = LET_GO.wait(h).unwrap_or_else(|e| e.into_inner());
        }
        let_go(&mut h.waiting, range);
    }
    h.remapping.push(range);
    drop(h);
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        REMAPS[0].fetch_add(1, Relaxed);
        REMAPS[1].fetch_add(asked.elapsed().as_micros() as u64, Relaxed);
    }
    Remapping {
        range,
        asked,
        #[cfg(feature = "diagnostics")]
        had: std::time::Instant::now(),
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(Vec<(usize, usize)>);

impl Drop for Reading {
    fn drop(&mut self) {
        let mut h = held();
        for &range in &self.0 {
            let_go(&mut h.reading, range);
        }
        drop(h);
        READINGS_HERE.set(READINGS_HERE.get() - 1);
        LET_GO.notify_all();
    }
}

std::thread_local! {
    /// Readings this thread holds ([`Reading`]): a change of that memory
    /// waits for this thread, which then must not wait for a change itself
    /// (`crate::pagemap`'s fault handler asks).
    static READINGS_HERE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
}

/// Whether this thread holds a reading of title memory in place
/// ([`title_memory_reading`]).
pub(crate) fn reading_here() -> bool {
    READINGS_HERE.get() != 0
}

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping there, so memory
/// found mapped stays mapped. Not to be taken twice by one thread (a
/// change waiting for the first would hold back the second: take both at
/// once, [`title_memory_reading_all`]), nor held across anything that maps
/// title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    title_memory_reading_all(&[(at, len)])
}

/// [`title_memory_reading`] for several ranges `(at, len)`, had together.
pub fn title_memory_reading_all(ranges: &[(usize, usize)]) -> Reading {
    let ranges: Vec<(usize, usize)> = ranges.iter().map(|&(at, len)| range(at, len)).collect();
    let mut h = held();
    while ranges.iter().any(|&r| meets(&h.remapping, r) || meets(&h.waiting, r)) {
        h = LET_GO.wait(h).unwrap_or_else(|e| e.into_inner());
    }
    h.reading.extend_from_slice(&ranges);
    READINGS_HERE.set(READINGS_HERE.get() + 1);
    Reading(ranges)
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping there (or waits to), for a caller that has
/// another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    let range = range(at, len);
    let mut h = held();
    if meets(&h.remapping, range) || meets(&h.waiting, range) {
        return None;
    }
    h.reading.push(range);
    READINGS_HERE.set(READINGS_HERE.get() + 1);
    Some(Reading(vec![range]))
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(3, std::time::Instant::now());
    {
        let mut log = REMAP_LOG.lock().unwrap_or_else(|e| e.into_inner());
        log.next += 1;
        let seq = log.next;
        if log.entries.len() >= REMAP_LOG_MOST {
            log.entries.pop_front();
        }
        log.entries.push_back((seq, at, at.saturating_add(len)));
    }
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = Held::new();
    let (from, to) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    // A check making blocks here read-only notes them after this, as when
    // it held the guards meanwhile ([`guard_unlocked`]).
    wait_protecting(&mut g, |g| g.protecting.count_in(from, to) != 0);
    g.absent.remove_range(from, to);
    forget_holes(&mut g, from, to);
    // (Clean blocks are some of the protected ones.)
    g.protected.remove_range(from, to);
    g.clean.remove_range(from, to);
    g.aliased.remove_range(from, to);
    g.rests.remapped(from, to);
    if view_known() {
        // The runtime calls this for two things only (`xmem`: `map` and
        // `unmap`): a range it has just mapped a section's pages into,
        // and a view it is about to make a placeholder again. Either way
        // what stands there is view memory or a hole in it, which is what
        // `is_view` would answer for a range that begins there; and the
        // write watch's asking can only fail there. The title remaps some
        // 83 times a frame, and each time the blocks were forgotten, so
        // that the next asking about each range went to the kernel twice
        // (run 235: 266 to 565 failed askings a frame with the blocks
        // remembered from their guards alone, no fewer than without).
        g.viewed.insert_range(from, to);
        return;
    }
    // What the range shows is new: no longer known to be a view.
    g.viewed.remove_range(from, to);
}

/// The runtime's changes of what title memory shows, in order
/// ([`title_memory_remapped`]): the number of the last, and the most recent
/// ones with their numbers, for [`remapped_since`].
struct RemapLog {
    next: u64,
    entries: std::collections::VecDeque<(u64, usize, usize)>,
}

static REMAP_LOG: std::sync::Mutex<RemapLog> =
    std::sync::Mutex::new(RemapLog { next: 0, entries: std::collections::VecDeque::new() });

/// The changes [`REMAP_LOG`] keeps (some 80 a frame in play).
const REMAP_LOG_MOST: usize = 16384;

/// The number of the runtime's last change of what title memory shows. Read
/// while holding [`title_memory_reading`] of a range: what is found mapped
/// there stays so until a later change.
pub(crate) fn remap_seq() -> u64 {
    REMAP_LOG.lock().unwrap_or_else(|e| e.into_inner()).next
}

/// Whether the runtime may have changed what `[at, at + len)` shows since
/// its change number `seq` ([`remap_seq`]): a later change that meets the
/// range, or one too old to be kept any more.
pub(crate) fn remapped_since(seq: u64, at: usize, len: usize) -> bool {
    let log = REMAP_LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.next == seq {
        return false;
    }
    if log.entries.front().is_none_or(|&(first, _, _)| first > seq + 1) {
        return true;
    }
    let end = at.saturating_add(len);
    log.entries.iter().rev().take_while(|&&(n, _, _)| n > seq).any(|&(_, from, to)| from < end && at < to)
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason). A
        // block being made read-only meanwhile is waited for (`open`): then
        // it is guarded, and opened.
        let mut g = Held::new();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The ahead ranges answer what a walk of them would, whatever order
    /// they come in, nested and overlapping ones among them.
    #[test]
    fn ahead_ranges_hold_what_a_walk_finds() {
        let ranges = [(50, 60), (10, 20), (15, 40), (70, 75), (12, 13), (0, 5), (55, 90), (30, 31)];
        let mut ahead = Ahead::new();
        for (n, &range) in ranges.iter().enumerate() {
            ahead.push(range);
            let added = &ranges[..=n];
            for first in 0..100 {
                for last in first + 1..=100 {
                    let walk = added.iter().any(|&(a, b)| a <= first && last <= b);
                    assert_eq!(ahead.holds(first, last), walk, "{first}..{last} after {n}");
                }
            }
        }
        ahead.clear();
        assert!(ahead.is_empty() && !ahead.holds(10, 11));
    }

    /// With `VICTORY_ASK_EXACT`: the refreshes of a submission's copies,
    /// merged as `Batch` merges them ([`apart`] splits; a step of ours that
    /// writes title memory flushes), each lie in one asking made of the
    /// submission's ranges ([`ahead_runs`]), so each is answered from the
    /// journal; and the runs' span counts what they cover.
    #[test]
    fn exact_refreshes_lie_in_one_asking() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % n
        };
        for trial in 0..300 {
            // Three copies' steps over 512 MB, short and long, near and far.
            let mut steps: Vec<(usize, usize, usize)> = Vec::new();
            for _ in 0..1 + next(60) {
                let copy = next(3);
                let at = copy * (512 << 20) / 3 + next(160 << 20);
                let len = 1 + if next(4) == 0 { next(24 << 20) } else { next(256 << 10) };
                steps.push((copy, at, at + len));
            }
            // As `Batch` merges them, flushed now and then.
            let mut pending: [Option<(usize, usize)>; 3] = [None; 3];
            let mut refreshes = Vec::new();
            for &(copy, a, e) in &steps {
                if next(10) == 0 {
                    refreshes.extend(pending.iter_mut().filter_map(Option::take));
                }
                pending[copy] = match pending[copy] {
                    Some((from, to)) if !apart(from, to, a, e) => Some((from.min(a), to.max(e))),
                    Some(old) => {
                        refreshes.push(old);
                        Some((a, e))
                    }
                    None => Some((a, e)),
                };
            }
            refreshes.extend(pending.iter().flatten());
            // The submission's asking, of all its ranges.
            let mut pieces: Vec<(usize, usize)> = steps.iter().map(|&(_, a, e)| (a / PAGE, e.div_ceil(PAGE))).collect();
            pieces.sort_unstable();
            let (runs, spanned, held) = ahead_runs(&pieces);
            let spans: Vec<(usize, usize)> =
                runs.iter().map(|&(i, k)| (pieces[i].0, pieces[i..k].iter().map(|p| p.1).max().unwrap())).collect();
            for &(from, to) in &refreshes {
                let (first, last) = (from / PAGE, to.div_ceil(PAGE));
                assert!(spans.iter().any(|&(a, b)| a <= first && last <= b), "trial {trial}: {from:#x}..{to:#x} in no asking");
            }
            assert!(held <= spanned);
            let asked: usize = runs.iter().zip(&spans).filter(|(r, _)| r.1 - r.0 >= 2).map(|(_, s)| s.1 - s.0).sum();
            assert_eq!(spanned, asked);
        }
    }

    /// A run of pieces in two reservations is asked by halves; the pages
    /// found come after what the answer held. Asked on the kept threads
    /// (`spread_own`), range by range, they are the same.
    #[test]
    fn kernel_runs_split_at_reservations() {
        let alloc = || {
            // SAFETY: a fresh write-watched allocation, freed at the end.
            let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
            assert_ne!(p, 0);
            p
        };
        let (a, b) = (alloc(), alloc());
        // Two pieces in each (a piece alone is left to `writes`).
        let (a, b) = (a.min(b), a.max(b));
        let pieces = [(a / PAGE, a / PAGE + 2), (a / PAGE + 2, a / PAGE + 4), (b / PAGE, b / PAGE + 2), (b / PAGE + 2, b / PAGE + 4)];
        let ranges = [(a / PAGE, a / PAGE + 4), (b / PAGE, b / PAGE + 4)];
        for wide in [false, true] {
            // SAFETY: committed pages of ours.
            unsafe {
                ((a + PAGE) as *mut u8).write_volatile(1);
                ((b + 2 * PAGE) as *mut u8).write_volatile(1);
            }
            let (mut asked, mut pages) = (Vec::new(), vec![7]);
            if wide {
                for found in crate::shadow::spread_own(2, &|r| {
                    let mut pages = Vec::new();
                    kernel_pages(ranges[r].0, ranges[r].1, &mut pages).then_some(pages)
                }) {
                    asked.push(());
                    pages.extend(found.expect("write-watched"));
                }
            } else {
                let mut run = Vec::new();
                kernel_run(&pieces, &mut run, &mut pages);
                assert_eq!(run, ranges);
                asked.extend(run.iter().map(|_| ()));
            }
            assert_eq!(asked.len(), 2, "wide: {wide}");
            assert_eq!(pages[0], 7);
            let mut found = pages[1..].to_vec();
            found.sort_unstable();
            assert_eq!(found, [a / PAGE + 1, b / PAGE + 2], "wide: {wide}");
        }
        for p in [a, b] {
            // SAFETY: our allocations.
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// `writes_after` lists what `writes` lists, less the pages last written
    /// at or before its asking: "nothing after the asking" is an empty list
    /// exactly where every write `writes` gives is that old (as the texture
    /// and page-copy checks ask it), for write-watched memory and for view
    /// memory the guards answer.
    #[test]
    fn writes_after_leaves_out_only_older_writes() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile};
        const PAGES: usize = 4;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let watched = unsafe { VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(watched, 0);
        // SAFETY: a new pagefile section, mapped whole; the view holds its
        // own reference once the handle is closed.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (PAGES * PAGE) as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize;
            windows_sys::Win32::Foundation::CloseHandle(section);
            view
        };
        assert_ne!(view, 0);
        for (at, guarded) in [(watched, false), (view, true)] {
            let range = (at, at + PAGES * PAGE);
            let (first, _) = writes(range.0, range.1, guarded).expect("watched or guarded");
            // Pages 1 and 3 written, each after an asking of its own.
            let mut askings = vec![first];
            for page in [1, 3] {
                // SAFETY: a byte of our memory (a guarded block faults and opens).
                unsafe { ((at + page * PAGE + 8) as *mut u8).write_volatile(1) };
                askings.push(writes(range.0, range.1, guarded).unwrap().0);
            }
            let (now, all) = writes(range.0, range.1, guarded).unwrap();
            // (A view's first check reports all its pages, at `first`.)
            let later: Vec<usize> = all.iter().filter(|w| w.1 > first).map(|w| w.0 - at / PAGE).collect();
            assert_eq!(later, [1, 3]);
            askings.push(now);
            for since in askings {
                let (_, after) = writes_after(range.0, range.1, guarded, since).unwrap();
                let older: Vec<(usize, u64)> = all.iter().copied().filter(|&(_, w)| w > since).collect();
                assert_eq!(after, older, "after asking {since} (guarded: {guarded})");
                // Into a buffer that held an answer before: the same list.
                let mut into = vec![(usize::MAX, u64::MAX); 3];
                assert!(writes_after_into(range.0, range.1, guarded, since, &mut into).is_some());
                assert_eq!(into, older, "into a buffer, after asking {since} (guarded: {guarded})");
                assert_eq!(after.is_empty(), all.iter().all(|&(_, w)| w <= since));
            }
        }
        forget_written(watched, watched + PAGES * PAGE);
        forget_written(view, view + PAGES * PAGE);
        title_memory_remapped(view, PAGES * PAGE);
        // SAFETY: our view and allocation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            VirtualFree(watched as *mut _, 0, MEM_RELEASE);
        }
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // Told while the view still holds the addresses: once it is unmapped
        // another test's view may be put there, and find these guards.
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 4 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        let untold = start + 3 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE, untold / PAGE]);
        // The last blocks are still entirely reserved.
        let again = refresh();
        assert!(!again.contains(&(late / PAGE)) && !again.contains(&(untold / PAGE)));

        // Ordinary commit, without a runtime mapping notification; told as
        // `xmem` tells its commits into reservations.
        commit(late, KERNEL_PAGE);
        title_memory_committed(late, KERNEL_PAGE);
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);

        // A commit nobody tells is seen once Windows' last word on the
        // block is older than `hole_keep`.
        commit(untold, KERNEL_PAGE);
        // SAFETY: inside the page just committed.
        unsafe { (untold as *mut u8).write_volatile(1) };
        std::thread::sleep(std::time::Duration::from_millis(hole_keep() + 20));
        assert!(refresh().contains(&(untold / PAGE)));
    }

    /// What is kept of Windows' answers about uncommitted blocks: trimmed
    /// around the blocks that may have been committed, replaced by a newer
    /// answer, and taken only while it is fresh.
    #[test]
    fn holes_are_kept_trimmed_and_expire() {
        let mut g = Guards::new();
        let spans = |g: &Guards| g.holes.iter().map(|(&s, h)| (s, h.end)).collect::<Vec<_>>();
        // Said at frame 7, millisecond 1000, and looked up a millisecond
        // later (the frame counter and the clock are the process's).
        let said = |g: &mut Guards| g.holes.values_mut().for_each(|h| (h.frame, h.ms) = (7, 1000));
        let known_hole = |g: &Guards, block| known_hole_at(g, block, 7, 1001, 100);
        note_hole(&mut g, 10, 20, false);
        note_hole(&mut g, 30, 40, false);
        said(&mut g);
        assert_eq!(known_hole(&g, 15), Some((20, false)));
        assert_eq!(known_hole(&g, 20), None, "past its end");
        assert_eq!(known_hole(&g, 5), None);
        // A mapping in the middle of the first and over the start of the second.
        forget_holes(&mut g, 12, 14);
        forget_holes(&mut g, 25, 32);
        assert_eq!(spans(&g), [(10, 12), (14, 20), (32, 40)]);
        assert_eq!(known_hole(&g, 13), None);
        assert_eq!(known_hole(&g, 11), Some((12, false)));
        // A newer answer replaces what overlaps it; a committed block (an
        // answer that ends where it begins) only takes the old word away.
        note_hole(&mut g, 16, 36, true);
        said(&mut g);
        assert_eq!(spans(&g), [(10, 12), (14, 16), (16, 36), (36, 40)]);
        assert_eq!(known_hole(&g, 33), Some((36, true)));
        note_hole(&mut g, 11, 11, false);
        assert_eq!(spans(&g), [(10, 11), (14, 16), (16, 36), (36, 40)]);
        assert_eq!(known_hole_at(&g, 33, 8, 1001, 100), Some((36, true)), "a frame later, still fresh");
        // Freshness: less than `keep` ms, whatever the frame.
        let hole = Hole { end: 1, mapped: false, frame: 7, ms: 1000 };
        assert!(fresh(&hole, 7, 1099, 100));
        assert!(!fresh(&hole, 7, 1100, 100));
        assert!(fresh(&hole, 8, 1001, 100), "a frame later");
        assert!(!fresh(&hole, 7, 1000, 0), "kept for no time at all");
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // Told while the placeholder still holds the addresses (another
        // test's view may be put there once it goes).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Told while the placeholder still holds the addresses (another
        // test's view may be put there once it goes).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// With the bits of absent blocks inside known holes ([`hole_bits`]),
    /// ranges holding unmapped blocks are answered as without them.
    #[test]
    fn unmapped_blocks_are_reported_with_hole_bits_as_without() {
        HOLE_BITS_HERE.set(true);
        unmapped_blocks_are_reported_once_until_mapped();
        a_range_that_begins_in_an_unmapped_block_is_guarded();
        HOLE_BITS_HERE.set(false);
    }

    /// With [`hole_bits`]: the absent blocks of a range inside a hole
    /// Windows said are quiet, and a check of clean and quiet blocks alone
    /// says nothing written; a commit takes a block out (it is reported
    /// once, then absent and quiet again), and so does the answer growing
    /// old (the next check finds the hole fresh and notes it again).
    #[test]
    fn absent_blocks_inside_known_holes_are_bits() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3,
            PAGE_NOACCESS, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        if hole_keep() == 0 {
            return;
        }
        HOLE_BITS_HERE.set(true);
        const BLOCKS: usize = 4;
        // SAFETY: a fresh placeholder of four blocks; its first block shows
        // a pagefile section's.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(GetCurrentProcess(), std::ptr::null(), BLOCKS * BLOCK, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let view = MapViewOfFile3(section, GetCurrentProcess(), p as *const _, 0, BLOCK, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert_eq!(view.Value as usize, p);
            (p, section, view)
        };
        let (b, pages) = (p / BLOCK, |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE);
        let quiet = |blocks: std::ops::Range<usize>| {
            let g = guards_shared();
            blocks.map(|k| g.quiet.contains(b + k)).collect::<Vec<bool>>()
        };
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some((0..BLOCKS).flat_map(pages).collect()));
        assert_eq!(quiet(0..BLOCKS), [false, true, true, true], "the absent blocks, not the mapped one");
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some(Vec::new()), "nothing written since");
        // Committed by someone who says so: reported once, then absent again.
        title_memory_committed(p + 2 * BLOCK, BLOCK);
        assert_eq!(quiet(0..BLOCKS), [false, true, false, true]);
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some(pages(2).collect()));
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some(Vec::new()));
        assert_eq!(quiet(0..BLOCKS), [false, true, true, true], "found in the hole again");
        // Every answer grown old: no block quiet until a check finds the
        // hole fresh again (the clock of the answers is the process's).
        quiet_expire(&mut guards(), u64::MAX / 2);
        assert_eq!(quiet(0..BLOCKS), [false; BLOCKS]);
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some(Vec::new()));
        assert_eq!(quiet(0..BLOCKS), [false, true, true, true]);
        // A mapping there: the block is new, reported, and no longer quiet.
        title_memory_remapped(p + 3 * BLOCK, BLOCK);
        assert_eq!(quiet(0..BLOCKS), [false, true, true, false]);
        assert_eq!(guard(p, p + BLOCKS * BLOCK), Some(pages(3).collect()));
        HOLE_BITS_HERE.set(false);
        title_memory_remapped(p, BLOCKS * BLOCK);
        assert_eq!(quiet(0..BLOCKS), [false; BLOCKS]);
        // SAFETY: our view and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..BLOCKS {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// What [`Rests`] says: a block rests only if a write opens it soon
    /// after a check made it read-only, for the rest's length, once per
    /// check; long runs are not noted; old notes go when there are many.
    #[test]
    fn rests_begin_just_after_a_check_and_end() {
        const REST: u64 = 1000;
        const CHECKS: u32 = 4;
        let mut r = Rests::new();
        r.armed(10, 12, 5_000, REST);
        assert!(r.opened(10, 5_400, REST, CHECKS), "opened soon after its check");
        assert!(r.rests(10, 5_401) && r.rests(10, 6_399));
        assert!(!r.rests(10, 6_400), "its rest is over");
        assert!(!r.rests(10, 6_000), "and forgotten");
        // At most `CHECKS` checks, however long the time left.
        r.armed(30, 31, 5_000, REST);
        assert!(r.opened(30, 5_001, REST, CHECKS));
        for n in 0..CHECKS {
            assert!(r.rests(30, 5_002), "check {n} of its rest");
        }
        assert!(!r.rests(30, 5_003), "its checks are spent");
        assert!(!r.rests(30, 5_004));
        assert!(!r.opened(11, 6_000, REST, CHECKS), "opened too long after its check");
        r.armed(11, 12, 7_000, REST);
        assert!(r.opened(11, 7_001, REST, CHECKS));
        assert!(!r.opened(11, 7_002, REST, CHECKS), "once per check");
        assert!(!r.opened(20, 0, REST, CHECKS), "never made read-only by a check");
        // A range guarded whole for the first time is not a block found written.
        r.armed(100, 101 + REST_RUN_MOST, 8_000, REST);
        assert!(!r.opened(100, 8_001, REST, CHECKS));
        r.armed(100, 100 + REST_RUN_MOST, 8_000, REST);
        assert!(r.opened(100 + REST_RUN_MOST - 1, 8_001, REST, CHECKS));
        r.armed(150, 151, 8_000, REST);
        r.remapped(100, 200);
        assert!(!r.rests(100 + REST_RUN_MOST - 1, 8_002), "mapped anew");
        assert!(!r.opened(150, 8_002, REST, CHECKS), "mapped anew since its check");
        // Many notes: the old ones go, a fresh one stays.
        let mut r = Rests::new();
        for block in 0..RESTS_MOST - 1 {
            r.armed(block, block + 1, 0, REST);
        }
        r.armed(RESTS_MOST, RESTS_MOST + 1, 2 * REST - 1, REST);
        r.armed(RESTS_MOST + 1, RESTS_MOST + 2, 2 * REST, REST);
        assert_eq!(r.armed.len(), 2);
        assert!(r.opened(RESTS_MOST, 2 * REST, REST, CHECKS));
        // Many rests: the ended ones go.
        let mut r = Rests::new();
        for block in 0..RESTS_MOST {
            r.armed(block, block + 1, 0, REST);
            assert!(r.opened(block, 1, REST, CHECKS));
        }
        r.armed(RESTS_MOST, RESTS_MOST + 1, 2 * REST, REST);
        assert!(r.opened(RESTS_MOST, 2 * REST + 1, REST, CHECKS));
        assert_eq!((r.until.len(), r.rested.len), (1, 1));
        assert!(!r.rests(0, 2 * REST + 2) && r.rests(RESTS_MOST, 2 * REST + 2));
    }

    /// With rests: a block the title writes just after its check is left
    /// open and reported written at every check of its rest, then guarded
    /// again at the first check after it (its checks spent, or its time
    /// over); a block written long after its check is guarded again at its
    /// next check, as without rests.
    #[test]
    fn a_block_written_just_after_its_check_rests_then_is_guarded_again() {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        const SIZE: usize = 2 * BLOCK;
        const CHECKS: u32 = 3;
        // Long enough that no rest of this test ends by itself.
        REST_HERE.set(600_000_000);
        REST_CHECKS_HERE.set(CHECKS);
        // SAFETY: a fresh pagefile section mapped once; the view and the
        // handle are let go at the end.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            (section as usize, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize)
        };
        assert_ne!(view, 0);
        let (block, pages) = (view / BLOCK, |b: usize| (view + b * BLOCK) / PAGE);
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]));
        assert!(!writable(view) && !writable(view + BLOCK));
        // SAFETY: a byte of our view; the fault opens its block.
        unsafe { ((view + 8) as *mut u8).write_volatile(1) };
        assert!(guards_shared().rests.rested.contains(block), "written just after its check");
        for n in 0..CHECKS as u8 {
            assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]), "reported at every check of its rest");
            assert!(writable(view), "left open");
            assert!(!writable(view + BLOCK), "the block beside it stays guarded");
            // SAFETY: as above; the block is open, nothing faults.
            unsafe { ((view + 16) as *mut u8).write_volatile(n) };
        }
        // Its checks are spent: the next guards it again.
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]));
        assert!(!writable(view), "guarded again");
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        // Written just after that check: it rests again, until its time is over.
        // SAFETY: as above; the fault opens it.
        unsafe { ((view + 8) as *mut u8).write_volatile(2) };
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]));
        assert!(writable(view), "left open again");
        guards().rests.until.insert(block, (0, CHECKS));
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]));
        assert!(!writable(view), "guarded again");
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        // Written long after its check (its note gone, as old ones go).
        guards().rests.armed.remove(&block);
        // SAFETY: as above; the fault opens it.
        unsafe { ((view + 8) as *mut u8).write_volatile(3) };
        assert!(!guards_shared().rests.rested.contains(block));
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]));
        assert!(!writable(view), "guarded again at once");
        assert_eq!(guard(view, view + SIZE), Some(Vec::new()));
        REST_HERE.set(0);
        REST_CHECKS_HERE.set(0);
        drop_views(section, &[view], SIZE);
    }

    /// With rests, two views of the same pages, each checked for a copy of
    /// its own: a write through either is reported at both, whichever of
    /// them rests (a view guarded again makes the other read-only too,
    /// which ends that one's rest), and once the rests are over nothing is
    /// left to report.
    #[test]
    fn no_write_through_another_view_is_lost_while_a_view_rests() {
        const SIZE: usize = BLOCK;
        REST_HERE.set(600_000_000);
        REST_CHECKS_HERE.set(4);
        let (section, one, two) = two_views(SIZE);
        let (p1, p2) = (one / PAGE, two / PAGE);
        assert_eq!(guard(one, one + SIZE), Some(vec![p1]));
        assert_eq!(guard(two, two + SIZE), Some(vec![p2]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        // SAFETY: the first view; the fault opens both.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(guards_shared().rests.rested.contains(one / BLOCK), "written just after its check");
        // Writes through either view, the views checked in either order.
        for round in 0..12usize {
            let at = if round % 2 == 0 { two } else { one };
            // SAFETY: a view of ours; one that is read-only faults and opens.
            unsafe { ((at + 8 + round) as *mut u8).write_volatile(round as u8) };
            let order = if round % 4 < 2 { [(one, p1), (two, p2)] } else { [(two, p2), (one, p1)] };
            for (view, page) in order {
                assert_eq!(guard(view, view + SIZE), Some(vec![page]), "round {round}, view {view:#x}");
            }
        }
        {
            let mut g = guards();
            for block in [one / BLOCK, two / BLOCK] {
                if g.rests.until.contains_key(&block) {
                    g.rests.until.insert(block, (0, 0));
                }
            }
        }
        // The rests over, the checks guard both again: nothing is left to
        // report, and both are read-only.
        for _ in 0..2 {
            guard(one, one + SIZE);
            guard(two, two + SIZE);
        }
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        assert!(!writable(one) && !writable(two));
        REST_HERE.set(0);
        REST_CHECKS_HERE.set(0);
        drop_views(section, &[one, two], SIZE);
    }

    /// Without rests (the switch off), a block written just after its check
    /// is guarded again at the next check, as it always was.
    #[test]
    fn without_rests_a_written_block_is_guarded_again_at_once() {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        if rest_mode() != 0 {
            return;
        }
        // SAFETY: as in the test above.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            (section as usize, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize)
        };
        assert_ne!(view, 0);
        assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]));
        // SAFETY: a byte of our view; the fault opens its block.
        unsafe { ((view + 8) as *mut u8).write_volatile(1) };
        assert!(!guards_shared().rests.rested.contains(view / BLOCK));
        assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]));
        assert!(!writable(view));
        assert_eq!(guard(view, view + BLOCK), Some(Vec::new()));
        drop_views(section, &[view], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // (addresses of this test's own: the lock is the process's)
        const AT: usize = 0x7000_0000_0000;
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(AT, 0x1_0000);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(AT + 0x8000, 0x10_0000);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(AT, 0x1_0000);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(AT + 0xfff0, 0x100);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    #[test]
    fn ranges_had_together_are_held_together() {
        const AT: usize = 0x7200_0000_0000;
        let both = super::title_memory_reading_all(&[(AT, 0x100), (AT + 0x10_0000, 0x100)]);
        let (done, wait) = mpsc::channel();
        std::thread::spawn(move || {
            let _changing = title_memory_remapping(AT + 0x10_0000, 0x1_0000);
            done.send(()).unwrap();
        });
        assert!(wait.recv_timeout(Duration::from_millis(100)).is_err(), "the second range changed under the read");
        drop(both);
        wait.recv_timeout(Duration::from_secs(5)).expect("the change went on once the reads let go");
        assert!(super::held().reading.iter().all(|&(a, _)| !(AT..AT + 0x20_0000).contains(&a)));
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
    };

    /// [`super::writes_as`] with its answer in a vector of its own.
    fn writes_as(start: usize, end: usize, guarded: bool, shared: bool, since: u64) -> Option<(u64, Vec<(usize, u64)>)> {
        let mut written = Vec::new();
        super::writes_as(start, end, guarded, shared, since, &mut written).map(|now| (now, written))
    }

    /// [`super::writes_shared`] with its answer in a vector of its own.
    fn writes_shared(first: usize, last: usize, guarded: bool, since: u64) -> Option<(u64, Vec<(usize, u64)>)> {
        let mut written = Vec::new();
        super::writes_shared(first, last, guarded, since, &mut written).map(|now| (now, written))
    }

    /// A pagefile view of `blocks` blocks, unmapped (and its guards
    /// forgotten) when dropped.
    struct View(usize, usize);

    impl View {
        fn new(blocks: usize) -> View {
            let size = blocks * BLOCK;
            // SAFETY: a new pagefile section, mapped whole; the view holds
            // its own reference once the handle is closed.
            let at = unsafe {
                let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, size as u32, std::ptr::null());
                assert!(!section.is_null());
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, size);
                CloseHandle(section);
                view.Value as usize
            };
            assert!(at != 0 && at.is_multiple_of(BLOCK));
            View(at, size)
        }
    }

    impl Drop for View {
        fn drop(&mut self) {
            // All told while the view still holds the addresses: once it is
            // unmapped another test's view may be put there.
            forget_written(self.0, self.0 + self.1);
            title_memory_remapped(self.0, self.1);
            // SAFETY: our view, still mapped.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.0 as *mut _ }) };
        }
    }

    fn guards_of(clean: &[usize], absent: &[usize]) -> Guards {
        Guards {
            protected: clean.iter().copied().collect(),
            clean: clean.iter().copied().collect(),
            absent: absent.iter().copied().collect(),
            ..Guards::new()
        }
    }

    /// Nothing to change: every block clean, or absent inside a hole known
    /// lately. Anything else is for `guard` to do.
    #[test]
    fn only_clean_blocks_and_known_holes_are_unchanged() {
        let none = |_: &Guards, _: usize| None;
        // Blocks 10 to 13 clean.
        let g = guards_of(&[10, 11, 12, 13], &[]);
        assert!(unchanged_with(&g, 10, 14, none));
        assert!(!unchanged_with(&g, 9, 14, none), "block 9 is neither clean nor absent");
        // Block 12 written (open): not clean any more.
        let g = guards_of(&[10, 11, 13], &[]);
        assert!(!unchanged_with(&g, 10, 14, none));
        // Blocks 11 and 12 absent, in a hole known to run to 13.
        let g = guards_of(&[10, 13], &[11, 12]);
        let hole = |_: &Guards, block: usize| (block == 11).then_some((13, false));
        assert!(unchanged_with(&g, 10, 14, hole));
        assert!(!unchanged_with(&g, 10, 14, none), "no hole known: Windows is asked");
        // A hole known to end at 12: block 12 is asked about again.
        let short = |_: &Guards, block: usize| (block == 11).then_some((12, false));
        assert!(!unchanged_with(&g, 10, 14, short));
    }

    /// Answers given side by side are those given alone: the same pages,
    /// a newer asking; a write the guards took sends the asking on alone,
    /// which reports it.
    #[test]
    fn answers_side_by_side_are_those_given_alone() {
        let view = View::new(8);
        let (start, end) = (view.0, view.0 + view.1);
        let (first, last) = (start / PAGE, end / PAGE);
        // The first asking guards the view (alone: nothing is known yet).
        assert!(writes_shared(first, last, true, 0).is_none());
        let (a, written) = writes_as(start, end, true, false, 0).expect("view memory");
        let (b, shared) = writes_shared(first, last, true, 0).expect("nothing to change");
        assert!(b > a && shared == written);
        // SAFETY: a byte of our view: the write faults and the guards open the block.
        unsafe { ((start + 3 * BLOCK + 40) as *mut u8).write_volatile(7) };
        assert!(writes_shared(first, last, true, 0).is_none(), "a block was written");
        let (c, after) = writes_as(start, end, true, true, 0).expect("view memory");
        assert!(c > b);
        assert!(after.contains(&(first + 3, c)), "the write is reported");
        let (d, again) = writes_shared(first, last, true, 0).expect("guarded again");
        assert!(d > c && again == after);
        // Memory that is no view: nothing side by side.
        let plain = vec![0u8; 3 * PAGE];
        let p = plain.as_ptr() as usize;
        assert!(writes_shared(p.div_ceil(PAGE), p.div_ceil(PAGE) + 1, true, 0).is_none());
    }

    /// Several threads keep copies of a view by the answers of their
    /// askings (as a refresh does: a page written after the copy took it is
    /// taken again) while another thread writes it: once the writes are
    /// done and each has asked once more, every copy holds the view's
    /// bytes, side by side or alone, and asking for the writes after their
    /// oldest page alone ([`writes_after`]).
    #[test]
    fn copies_kept_side_by_side_miss_no_write() {
        const BLOCKS: usize = 16;
        // (`unlocked`: the keepers' checks let the guards go while they make
        // blocks read-only, [`guard_unlocked`].)
        let cases = [(true, false, false), (false, false, false), (true, true, false), (false, true, false), (false, false, true), (true, true, true)];
        for (shared, after, unlocked) in cases {
            let view = View::new(BLOCKS);
            let (start, end) = (view.0, view.0 + view.1);
            assert!(writes_as(start, end, true, false, 0).is_some());
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let keep = |done: std::sync::Arc<std::sync::atomic::AtomicBool>| {
                move || {
                    UNLOCKED_HERE.set(unlocked);
                    let mut copy = vec![0u8; BLOCKS * BLOCK];
                    let mut copied = [0u64; BLOCKS];
                    let round = |copy: &mut Vec<u8>, copied: &mut [u64; BLOCKS]| {
                        // As `Shadow::taken_since` has it.
                        let since = match after {
                            true => copied.iter().copied().filter(|&at| at != 0).min().unwrap_or(u64::MAX),
                            false => 0,
                        };
                        let (now, written) = writes_as(start, end, true, shared, since).expect("view memory");
                        for (i, at) in copied.iter_mut().enumerate() {
                            let page = start / PAGE + i;
                            let write = written.iter().find(|w| w.0 == page).map_or(0, |w| w.1);
                            if *at == 0 || write > *at || *at < since {
                                // SAFETY: a block of our view, read only.
                                unsafe {
                                    std::ptr::copy_nonoverlapping((start + i * BLOCK) as *const u8, copy[i * BLOCK..].as_mut_ptr(), BLOCK)
                                };
                                *at = now;
                            }
                        }
                    };
                    while !done.load(std::sync::atomic::Ordering::SeqCst) {
                        round(&mut copy, &mut copied);
                    }
                    round(&mut copy, &mut copied);
                    copy
                }
            };
            let keepers: Vec<_> = (0..4).map(|_| std::thread::spawn(keep(done.clone()))).collect();
            let writer = std::thread::spawn(move || {
                for i in 0..20_000usize {
                    let at = start + (i * 7 % BLOCKS) * BLOCK + (i * 131) % BLOCK;
                    // SAFETY: a byte of our view; a guarded block faults and opens.
                    unsafe { (at as *mut u8).write_volatile((i % 251) as u8 + 1) };
                    if i % 64 == 0 {
                        std::thread::yield_now();
                    }
                }
            });
            writer.join().unwrap();
            done.store(true, std::sync::atomic::Ordering::SeqCst);
            // SAFETY: the whole view, read only.
            let bytes = unsafe { std::slice::from_raw_parts(start as *const u8, end - start) }.to_vec();
            for keeper in keepers {
                let copy = keeper.join().unwrap();
                let wrong = copy.iter().zip(&bytes).position(|(a, b)| a != b);
                assert!(wrong.is_none(), "a copy missed a write at {wrong:?} (side by side: {shared}, after: {after}, unlocked: {unlocked})");
            }
        }
    }
}

#[cfg(test)]
mod unlocked_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect,
    };

    /// A pagefile view of `blocks` blocks, guarded whole; unmapped (and
    /// forgotten) when dropped.
    struct View(usize, usize);

    impl View {
        fn guarded(blocks: usize) -> View {
            let size = blocks * BLOCK;
            // SAFETY: a new pagefile section, mapped whole; the view holds
            // its own reference once the handle is closed.
            let at = unsafe {
                let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, size as u32, std::ptr::null());
                assert!(!section.is_null());
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, size);
                CloseHandle(section);
                view.Value as usize
            };
            assert!(at != 0 && at.is_multiple_of(BLOCK));
            assert!(guard(at, at + size).is_some());
            View(at, size)
        }
    }

    impl Drop for View {
        fn drop(&mut self) {
            forget_written(self.0, self.0 + self.1);
            title_memory_remapped(self.0, self.1);
            // SAFETY: our view, still mapped.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.0 as *mut _ }) };
        }
    }

    /// As `protect` does with the lock let go: `block` put in `protecting`
    /// and made read-only. [`noted`] ends it.
    fn protecting(block: usize) {
        guards().protecting.insert(block);
        let mut old = 0;
        // SAFETY: a block of a view of ours; only its protection changes.
        assert_ne!(unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) }, 0);
    }

    /// As `protect` ends: `block` noted guarded and clean, out of
    /// `protecting`, and the waiters woken.
    fn noted(block: usize) {
        let mut g = guards();
        protect_noted(&mut g, block, block + 1, true, &[]);
        g.protecting.remove(block);
        *PROTECTED.lock().unwrap() += 1;
        PROTECTED_MOVED.notify_all();
    }

    /// Runs `f` on a thread of its own; whether it is done.
    fn spawned(f: impl FnOnce() + Send + 'static) -> (std::thread::JoinHandle<()>, Arc<AtomicBool>) {
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        let thread = std::thread::spawn(move || {
            f();
            flag.store(true, SeqCst);
        });
        (thread, done)
    }

    /// A write faulting on a block being made read-only, not yet noted,
    /// waits in the handler (it is not left to crash) until the block is
    /// noted, then opens it: the write lands and the block counts written.
    #[test]
    fn a_write_to_a_block_being_guarded_waits_then_counts() {
        let view = View::guarded(2);
        let block = view.0 / BLOCK;
        // Written: opened by the handler, as the walk finds it.
        // SAFETY: a byte of our view; the guarded block faults and opens.
        unsafe { (view.0 as *mut u8).write_volatile(1) };
        assert!(!guards_shared().protected.contains(block));
        protecting(block);
        let at = view.0 + 100;
        // SAFETY: as above; this time the block is read-only, not guarded.
        let (writer, done) = spawned(move || unsafe { (at as *mut u8).write_volatile(7) });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!done.load(SeqCst), "the write waits for the block to be noted");
        noted(block);
        writer.join().unwrap();
        // SAFETY: our view.
        assert_eq!(unsafe { (at as *const u8).read_volatile() }, 7);
        let g = guards_shared();
        assert!(!g.clean.contains(block) && !g.protected.contains(block), "opened and written");
        drop(g);
        assert_eq!(guard(view.0, view.0 + view.1), Some(vec![view.0 / PAGE]));
    }

    /// The runtime's steps wait for a block being made read-only too: a
    /// kernel write told ahead (`writing`) finds it open afterwards, and a
    /// remapping forgets its guard after it was noted, not before.
    #[test]
    fn writing_and_remapping_wait_for_a_block_being_guarded() {
        let view = View::guarded(2);
        let (first, second) = (view.0 / BLOCK, view.0 / BLOCK + 1);
        for block in [first, second] {
            // SAFETY: a byte of our view; the guarded block faults and opens.
            unsafe { ((block * BLOCK) as *mut u8).write_volatile(1) };
        }
        protecting(first);
        protecting(second);
        let at = view.0;
        let (told, writing_done) = spawned(move || writing(at + 8, 16));
        let (remapped, remapping_done) = spawned(move || title_memory_remapped(at + BLOCK, BLOCK));
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!writing_done.load(SeqCst) && !remapping_done.load(SeqCst), "both wait");
        noted(first);
        noted(second);
        told.join().unwrap();
        remapped.join().unwrap();
        assert!(writable(at + 8), "opened for the kernel's write");
        let g = guards_shared();
        assert!(!g.protected.contains(first) && !g.protected.contains(second), "opened; forgotten by the remapping");
        assert!(!g.protecting.contains(first) && !g.protecting.contains(second));
        drop(g);
        // The second block stays read-only with its guard forgotten (as after
        // a remapping that leaves the old view): ours again for the drop.
        let mut old = 0;
        // SAFETY: a block of a view of ours; only its protection changes.
        unsafe { VirtualProtect((second * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
}

#[cfg(test)]
mod askings_tests {
    use super::*;

    /// A page near the boundary between chunks 2 and 3.
    const EDGE: usize = 3 * ASK_CHUNK;

    #[test]
    fn pages_written_after_an_asking() {
        let mut j = Askings::new();
        assert!(j.after(0, usize::MAX >> 16, 0).is_empty());
        assert_eq!(j.set(EDGE - 1, 4), 0);
        assert_eq!(j.set(EDGE, 7), 0);
        assert_eq!(j.set(EDGE + 64, 9), 0);
        assert_eq!(j.set(EDGE, 8), 7, "the asking noted before");
        assert_eq!(j.get(EDGE), 8);
        assert_eq!(j.get(EDGE + 1), 0);
        assert_eq!(j.chunks.len(), 2);
        assert_eq!(j.after(EDGE - 10, EDGE + 100, 0), [(EDGE - 1, 4), (EDGE, 8), (EDGE + 64, 9)]);
        assert_eq!(j.after(EDGE - 10, EDGE + 100, 4), [(EDGE, 8), (EDGE + 64, 9)]);
        assert_eq!(j.after(EDGE - 10, EDGE + 100, 8), [(EDGE + 64, 9)]);
        assert!(j.after(EDGE - 10, EDGE + 100, 9).is_empty());
        assert_eq!(j.after(EDGE, EDGE + 64, 0), [(EDGE, 8)], "the range's end is its own");
        assert!(j.after(EDGE + 1, EDGE + 64, 0).is_empty());
        assert!(j.after(EDGE, EDGE, 0).is_empty());
        // Noted lower later (an asking that stood for a submission): the
        // group's highest stays, the page's own asking answers.
        j.set(EDGE + 64, 2);
        assert_eq!(j.after(EDGE + 64, EDGE + 65, 1), [(EDGE + 64, 2)]);
        assert!(j.after(EDGE + 64, EDGE + 65, 2).is_empty());
        j.forget(EDGE - 1, EDGE + 1);
        assert_eq!(j.after(0, usize::MAX >> 16, 0), [(EDGE + 64, 2)]);
    }

    /// Random notes and questions, answered as the `BTreeMap` the journal
    /// was answers them, filtered by the asking.
    #[test]
    fn answers_as_a_btree_map() {
        let mut state: u64 = 0x853c_49e6_748f_ea9b;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize
        };
        let bases = [0, ASK_CHUNK - 100, 5 * ASK_CHUNK - 30, (1 << 36) - 50];
        let (mut j, mut reference) = (Askings::new(), BTreeMap::new());
        for round in 0..20_000 {
            let page = bases[next() % bases.len()] + next() % 200;
            let asking = 1 + (next() % 1_000) as u64;
            assert_eq!(j.set(page, asking), reference.insert(page, asking).unwrap_or(0));
            let a = bases[next() % bases.len()] + next() % 200;
            let b = a + next() % 300;
            let since = (next() % 1_100) as u64;
            let expected: Vec<(usize, u64)> = reference.range(a..b).map(|(&p, &w)| (p, w)).filter(|&(_, w)| w > since).collect();
            assert_eq!(j.after(a, b, since), expected, "round {round}: [{a}, {b}) after {since}");
        }
    }
}

#[cfg(test)]
mod blocks_tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A few blocks either side of the boundary between chunks 1 and 2.
    const EDGE: usize = 2 * CHUNK;

    #[test]
    fn ranges_across_a_chunk_boundary() {
        let mut set = Blocks::new();
        set.insert_range(EDGE - 3, EDGE + 5);
        assert_eq!(set.len, 8);
        assert_eq!(set.chunks.len(), 2);
        assert!(set.all_in(EDGE - 3, EDGE + 5));
        assert!(!set.all_in(EDGE - 4, EDGE + 5));
        assert!(!set.all_in(EDGE - 3, EDGE + 6));
        assert!(set.all_in(EDGE, EDGE), "an empty range");
        assert_eq!(set.count_in(0, usize::MAX >> 16), 8);
        assert_eq!(set.count_in(EDGE - 1, EDGE + 1), 2);
        assert_eq!(set.iter_in(EDGE - 10, EDGE + 10).collect::<Vec<_>>(), (EDGE - 3..EDGE + 5).collect::<Vec<_>>());
        assert_eq!(set.iter_in(EDGE - 1, EDGE + 2).collect::<Vec<_>>(), [EDGE - 1, EDGE, EDGE + 1]);
        assert_eq!(set.first_unset(EDGE - 3, EDGE + 10), EDGE + 5);
        assert_eq!(set.first_unset(EDGE - 10, EDGE + 10), EDGE - 10);
        assert_eq!(set.first_unset(EDGE - 3, EDGE + 5), EDGE + 5, "all set: the range's end");
        // Inserting what is there already counts nothing twice.
        set.insert_range(EDGE - 1, EDGE + 1);
        assert!(!set.insert(EDGE));
        assert_eq!(set.len, 8);
        // Clearing across the boundary.
        set.remove_range(EDGE - 1, EDGE + 2);
        assert_eq!(set.iter_in(0, EDGE + CHUNK).collect::<Vec<_>>(), [EDGE - 3, EDGE - 2, EDGE + 2, EDGE + 3, EDGE + 4]);
        assert_eq!(set.len, 5);
        assert_eq!(set.first_unset(EDGE - 3, EDGE + 5), EDGE - 1);
    }

    #[test]
    fn empty_and_full_chunks() {
        let mut set = Blocks::new();
        assert!(set.is_empty());
        assert!(!set.contains(0) && !set.contains(usize::MAX >> 16));
        assert!(!set.all_in(5, 6));
        assert_eq!(set.first_unset(5, 9), 5);
        assert_eq!(set.count_in(0, usize::MAX >> 16), 0);
        // A whole chunk set, and the first block of the next.
        set.insert_range(CHUNK, 2 * CHUNK + 1);
        assert_eq!(set.count_in(0, 3 * CHUNK), CHUNK + 1);
        assert!(set.all_in(CHUNK, 2 * CHUNK + 1));
        assert_eq!(set.first_unset(CHUNK + 7, 3 * CHUNK), 2 * CHUNK + 1);
        // Emptied: its chunks stay, nothing in them is set.
        set.remove_range(0, usize::MAX >> 16);
        assert!(set.is_empty());
        assert_eq!(set.chunks.len(), 2);
        assert!(!set.contains(CHUNK) && !set.all_in(CHUNK, CHUNK + 1));
        assert_eq!(set.first_unset(CHUNK, 2 * CHUNK), CHUNK);
        assert_eq!(set.iter_in(0, usize::MAX >> 16).count(), 0);
        assert!(set.insert(CHUNK + 64));
        assert_eq!(set.iter_in(0, usize::MAX >> 16).collect::<Vec<_>>(), [CHUNK + 64]);
    }

    #[test]
    fn single_blocks_and_removal() {
        let mut set: Blocks = [3, 64, 65, 127, 128, CHUNK - 1, CHUNK].into_iter().collect();
        assert_eq!(set.len, 7);
        assert!(set.contains(64) && set.contains(CHUNK - 1) && !set.contains(66));
        assert!(set.remove(64));
        assert!(!set.remove(64), "removed already");
        assert!(!set.remove(5 * CHUNK), "a chunk never made");
        assert_eq!(set.iter_in(0, 2 * CHUNK).collect::<Vec<_>>(), [3, 65, 127, 128, CHUNK - 1, CHUNK]);
        assert_eq!(set.count_in(65, 128), 2);
        assert!(set.all_in(127, 129));
        set.remove_range(100, CHUNK);
        assert_eq!(set.iter_in(0, 2 * CHUNK).collect::<Vec<_>>(), [3, 65, CHUNK]);
        // Blocks at the top of the address space.
        let top = usize::MAX.div_ceil(BLOCK) - 1;
        assert!(set.insert(top));
        assert!(set.all_in(top, top + 1) && set.count_in(top, top + 1) == 1);
        set.remove_range(top, usize::MAX.div_ceil(BLOCK));
        assert!(!set.contains(top));
        assert_eq!(set.len, 3);
    }

    /// `insert_and` sets what the other set holds in the range, across
    /// chunks, and counts only blocks new to it.
    #[test]
    fn the_blocks_another_set_holds_in_a_range() {
        let other: Blocks = [3, 64, 65, CHUNK - 1, CHUNK, CHUNK + 70, 3 * CHUNK].into_iter().collect();
        let mut set: Blocks = [65, 2 * CHUNK].into_iter().collect();
        assert_eq!(set.insert_and(&other, 4, CHUNK + 70), 3, "64, CHUNK - 1 and CHUNK; 65 was set");
        assert_eq!(set.iter_in(0, 4 * CHUNK).collect::<Vec<_>>(), [64, 65, CHUNK - 1, CHUNK, 2 * CHUNK]);
        assert_eq!(set.len, 5);
        assert_eq!(set.insert_and(&other, 0, 4 * CHUNK), 3, "3, CHUNK + 70, 3 * CHUNK");
        assert_eq!(set.insert_and(&other, 0, 4 * CHUNK), 0);
        assert_eq!(set.insert_and(&Blocks::new(), 0, 4 * CHUNK), 0);
        assert_eq!(set.len, 8);
    }

    /// A small generator of numbers for the tests below.
    fn next(state: &mut u64) -> usize {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as usize
    }

    /// Random changes and questions, answered as a `BTreeSet` answers them,
    /// over blocks near chunk boundaries and far apart.
    #[test]
    fn answers_as_a_btree_set() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let bases = [0, CHUNK - 200, 7 * CHUNK - 100, (1 << 33) - 300];
        let block = |state: &mut u64| bases[next(state) % bases.len()] + next(state) % 600;
        let (mut set, mut reference) = (Blocks::new(), BTreeSet::new());
        for round in 0..20_000 {
            let (a, b) = (block(&mut state), block(&mut state));
            let (from, to) = (a.min(b), a.max(b) + 1);
            match next(&mut state) % 6 {
                0 => assert_eq!(set.insert(a), reference.insert(a)),
                1 => assert_eq!(set.remove(a), reference.remove(&a)),
                2 if to - from < 2_000 => {
                    set.insert_range(from, to);
                    reference.extend(from..to);
                }
                3 => {
                    set.remove_range(from, to);
                    reference.retain(|x| !(from..to).contains(x));
                }
                _ => {}
            }
            assert_eq!(set.len, reference.len(), "round {round}");
            assert_eq!(set.contains(b), reference.contains(&b));
            let within: Vec<usize> = reference.range(from..to).copied().collect();
            assert_eq!(set.iter_in(from, to).collect::<Vec<_>>(), within, "round {round}");
            assert_eq!(set.count_in(from, to), within.len());
            assert_eq!(set.all_in(from, to), within.len() == to - from);
            let unset = (from..to).find(|x| !reference.contains(x)).unwrap_or(to);
            assert_eq!(set.first_unset(from, to), unset, "round {round}: [{from}, {to})");
        }
    }

    /// `next_open` finds the block `guard`'s walk looked at block by block:
    /// the first neither clean nor absent inside the hole.
    #[test]
    fn the_walk_stops_where_it_did_block_by_block() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        for _ in 0..2_000 {
            let base = CHUNK - 150 + next(&mut state) % 100;
            let mut g = Guards::new();
            for block in base..base + 300 {
                match next(&mut state) % 8 {
                    0..4 => {
                        g.clean.insert(block);
                    }
                    4..6 => {
                        g.absent.insert(block);
                    }
                    _ => {}
                }
            }
            let at = base + next(&mut state) % 300;
            let last = at + next(&mut state) % (base + 301 - at);
            let hole_until = base + next(&mut state) % 320;
            let expected = (at..last)
                .find(|&b| !g.clean.contains(b) && !(g.absent.contains(b) && b < hole_until))
                .unwrap_or(last);
            assert_eq!(next_open(&g, at, hole_until, last), expected, "[{at}, {last}) hole to {hole_until}");
        }
    }

    /// What the walk reads ahead of the holes answers as the map does, past
    /// the holes read at once and across changes of the map.
    #[test]
    fn holes_read_ahead_answer_as_the_map() {
        let mut state = 0x6c07_8965_9e37_79b9;
        for _ in 0..300 {
            let mut g = Guards::new();
            let (base, last) = (1_000, 3_000);
            let mut b = base;
            while b < last {
                let len = 1 + next(&mut state) % 4;
                if !next(&mut state).is_multiple_of(3) {
                    note_hole(&mut g, b, b + len, next(&mut state).is_multiple_of(2));
                }
                b += len + next(&mut state) % 5;
            }
            let mut ahead = HolesAhead::new(last);
            let mut block = base - 5 + next(&mut state) % 10;
            while block < last {
                // A new answer of Windows meanwhile (or a block found committed).
                if next(&mut state).is_multiple_of(40) {
                    let at = block + next(&mut state) % 100;
                    note_hole(&mut g, at, at + next(&mut state) % 3, false);
                    ahead.asked();
                }
                assert_eq!(ahead.known(&g, block), known_hole(&g, block), "block {block}");
                block += 1 + next(&mut state) % 6;
            }
        }
    }
}

/// The time of a frame's checks of guarded view memory with nothing written
/// ([`guard`]): run AM's frame asks about some 150,000 blocks in some 2,000
/// checks, nearly all clean. A timing, run by hand with optimized code:
/// `cargo test --profile diag -p victory-gpu --lib --features diagnostics
/// watch::bench_tests -- --ignored --nocapture --test-threads=1`.
#[cfg(test)]
mod bench_tests {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
        MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `rounds` frames of 2,000 checks of `len` blocks each, spread over the
    /// `blocks` blocks at `at`: the time of one frame.
    fn frame_of_checks(at: usize, blocks: usize, len: usize, rounds: u32) -> std::time::Duration {
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            for k in 0..2000 {
                let first = (k * 37) % (blocks - len + 1);
                let written = guard(at + first * BLOCK, at + (first + len) * BLOCK).expect("view memory");
                assert!(written.is_empty(), "nothing written");
            }
        }
        started.elapsed() / rounds
    }

    #[test]
    #[ignore = "a timing, run by hand"]
    fn a_frame_of_checks_of_clean_blocks() {
        // One view of 2,048 blocks (128 MiB), guarded whole.
        const BLOCKS: usize = 2048;
        let size = BLOCKS * BLOCK;
        // SAFETY: a new pagefile section, mapped whole.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, size as u32, std::ptr::null());
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, size).Value as usize)
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + size).is_some());
        let clean = frame_of_checks(view, BLOCKS, 75, 50);
        let whole = frame_of_checks(view, BLOCKS, 2000, 2);

        // A placeholder of 512 blocks with every other block mapped: the
        // others are noted absent, each a hole of its own.
        const HOLEY: usize = 512;
        // SAFETY: a fresh placeholder and a pagefile section of its size.
        let (p, holey_section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                HOLEY * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let s = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, (HOLEY * BLOCK) as u32, std::ptr::null());
            assert!(!s.is_null());
            (p, s)
        };
        let views: Vec<MEMORY_MAPPED_VIEW_ADDRESS> = (0..HOLEY)
            .step_by(2)
            .map(|block| {
                // SAFETY: splits the placeholder at the block and maps the
                // section's block there.
                let view = unsafe {
                    VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                    MapViewOfFile3(
                        holey_section,
                        GetCurrentProcess(),
                        (p + block * BLOCK) as *const _,
                        (block * BLOCK) as u64,
                        BLOCK,
                        MEM_REPLACE_PLACEHOLDER,
                        PAGE_READWRITE,
                        std::ptr::null_mut(),
                        0,
                    )
                };
                assert_eq!(view.Value as usize, p + block * BLOCK);
                view
            })
            .collect();
        assert!(guard(p, p + HOLEY * BLOCK).is_some());
        assert_eq!(guard(p, p + HOLEY * BLOCK), Some(Vec::new()));
        let holey = frame_of_checks(p, HOLEY, 75, 10);
        println!(
            "a frame of checks with nothing written: 2,000 of 75 clean blocks {clean:?}; 2,000 of 2,000 clean blocks {whole:?}; \
             2,000 of 75 blocks every other one unmapped {holey:?}"
        );

        title_memory_remapped(view, size);
        title_memory_remapped(p, HOLEY * BLOCK);
        // SAFETY: our views, placeholders and sections.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            CloseHandle(section);
            for view in views {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..HOLEY {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            CloseHandle(holey_section);
        }
    }

    /// A frame's answers of the journal: some 20,000 refreshes of 14 pages
    /// on average (run AM: 19,600 refreshes, 18 GB considered), in memory
    /// whose every page was written once; as the `BTreeMap` gave them (the
    /// range read whole into a vector), and as [`Askings::after`] gives
    /// them to copies that took their pages after those writes.
    #[test]
    #[ignore = "a timing, run by hand"]
    fn a_frame_of_journal_answers() {
        const PAGES: usize = 280_000;
        let (mut map, mut j) = (BTreeMap::new(), Askings::new());
        let base = 0x1_0000_0000 / PAGE;
        for i in 0..PAGES {
            let asking = 1 + (i * 7919 % 1000) as u64;
            map.insert(base + i, asking);
            j.set(base + i, asking);
        }
        let ranges: Vec<(usize, usize)> = (0..20_000).map(|k| (base + k * 14, base + k * 14 + 14)).collect();
        let time = |f: &dyn Fn(usize, usize) -> usize| {
            let started = std::time::Instant::now();
            let mut found = 0;
            for _ in 0..10 {
                for &(a, b) in &ranges {
                    found += f(a, b);
                }
            }
            (started.elapsed() / 10, found / 10)
        };
        let tree = time(&|a, b| map.range(a..b).map(|(&p, &w)| (p, w)).collect::<Vec<_>>().len());
        let all = time(&|a, b| j.after(a, b, 0).len());
        let newer = time(&|a, b| j.after(a, b, 1000).len());
        println!("a frame of journal answers: as the tree gave them {tree:?}; all of them now {all:?}; those after the copies' askings {newer:?}");
    }

    /// How long a write of another thread's waits in the fault handler while
    /// a check guards many runs again (one `VirtualProtect` each), with the
    /// guards held through the check and let go around its system calls
    /// ([`guard_unlocked`]).
    #[test]
    #[ignore = "a timing, run by hand"]
    fn faults_beside_a_check_that_guards_again() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        // SAFETY: a new pagefile section of `blocks` blocks, mapped whole.
        let map = |blocks: usize| unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, (blocks * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value as usize;
            CloseHandle(section);
            assert_ne!(at, 0);
            at
        };
        const CHECKED: usize = 512;
        const WRITTEN: usize = 64;
        for unlocked in [false, true] {
            let (checked, written) = (map(CHECKED), map(WRITTEN));
            assert!(guard(checked, checked + CHECKED * BLOCK).is_some());
            assert!(guard(written, written + WRITTEN * BLOCK).is_some());
            let done = std::sync::Arc::new(AtomicBool::new(false));
            let checker = {
                let done = done.clone();
                std::thread::spawn(move || {
                    UNLOCKED_HERE.set(unlocked);
                    for _ in 0..40 {
                        // Every other block written: 256 runs to guard again.
                        for block in (0..CHECKED).step_by(2) {
                            // SAFETY: a byte of our view; a guarded block faults and opens.
                            unsafe { ((checked + block * BLOCK) as *mut u8).write_volatile(1) };
                        }
                        assert!(guard(checked, checked + CHECKED * BLOCK).is_some());
                    }
                    done.store(true, SeqCst);
                })
            };
            let (mut faults, mut total, mut longest) = (0u32, std::time::Duration::ZERO, std::time::Duration::ZERO);
            let mut block = 0;
            while !done.load(SeqCst) {
                let at = written + block * BLOCK;
                let started = std::time::Instant::now();
                // SAFETY: a byte of our view, guarded: it faults and opens.
                unsafe { (at as *mut u8).write_volatile(2) };
                let took = started.elapsed();
                (faults, total, longest) = (faults + 1, total + took, longest.max(took));
                block = (block + 1) % WRITTEN;
                if block == 0 {
                    assert!(guard(written, written + WRITTEN * BLOCK).is_some());
                }
            }
            checker.join().unwrap();
            println!(
                "writes beside a check guarding 256 runs again (lock let go: {unlocked}): {faults} writes, {:?} on average, {longest:?} at longest",
                total / faults.max(1)
            );
            for (at, blocks) in [(checked, CHECKED), (written, WRITTEN)] {
                title_memory_remapped(at, blocks * BLOCK);
                // SAFETY: our view.
                unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ }) };
            }
        }
    }

    /// By hand: a thread keeps writing 32 of a range's 256 blocks (each
    /// again some 0.5 ms later, as the title writes what it streams), while
    /// another checks the range every 250 us (the refreshes) and 16 more
    /// threads of the process run (the title's, whose TLBs a change of
    /// protection must shoot down). Without and with rests ([`rest_now`]):
    /// the time the checks took, the blocks they made read-only and the
    /// `VirtualProtect` calls, and the writes' faults.
    #[test]
    #[ignore = "a timing, run by hand"]
    fn checks_beside_a_writer_with_and_without_rests() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        const BLOCKS: usize = 256;
        const HOT: usize = 32;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let spinners: Vec<_> = (0..16)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut v = vec![0u8; 1 << 16];
                    let mut i = 0usize;
                    while !stop.load(SeqCst) {
                        v[i & 0xffff] = v[i & 0xffff].wrapping_add(1);
                        i = i.wrapping_add(4099);
                    }
                })
            })
            .collect();
        for (pause, rest, most) in [(15, 0, 1), (15, 2000, 2), (15, 4000, 4), (120, 0, 1), (120, 2000, 2), (120, 4000, 4)] {
            // SAFETY: a new pagefile section, mapped whole.
            let at = unsafe {
                let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * BLOCK) as u32, std::ptr::null());
                assert!(!section.is_null());
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCKS * BLOCK).Value as usize;
                CloseHandle(section);
                assert_ne!(at, 0);
                at
            };
            assert!(guard(at, at + BLOCKS * BLOCK).is_some());
            let done = std::sync::Arc::new(AtomicBool::new(false));
            let writer = {
                let done = done.clone();
                std::thread::spawn(move || {
                    REST_HERE.set(rest);
                    REST_CHECKS_HERE.set(most);
                    let (mut writes, mut slow, mut slow_time) = (0u64, 0u64, std::time::Duration::ZERO);
                    let mut n = 0;
                    while !done.load(SeqCst) {
                        let block = (n % HOT) * (BLOCKS / HOT);
                        let started = std::time::Instant::now();
                        // SAFETY: a byte of our view; a guarded block faults and opens.
                        unsafe { ((at + block * BLOCK + 8) as *mut u8).write_volatile(n as u8) };
                        let took = started.elapsed();
                        // A write that took a microsecond or more faulted.
                        if took >= std::time::Duration::from_micros(1) {
                            (slow, slow_time) = (slow + 1, slow_time + took);
                        }
                        writes += 1;
                        n += 1;
                        let waited = std::time::Instant::now();
                        while waited.elapsed() < std::time::Duration::from_micros(pause) {}
                    }
                    (writes, slow, slow_time)
                })
            };
            REST_HERE.set(rest);
            REST_CHECKS_HERE.set(most);
            for c in &RESTED {
                c.store(0, SeqCst);
            }
            let (mut checks, mut checking, mut reported) = (0u32, std::time::Duration::ZERO, 0usize);
            let started = std::time::Instant::now();
            while started.elapsed() < std::time::Duration::from_secs(2) {
                let check = std::time::Instant::now();
                reported += guard(at, at + BLOCKS * BLOCK).expect("a view").len();
                checking += check.elapsed();
                checks += 1;
                while check.elapsed() < std::time::Duration::from_micros(250) {}
            }
            done.store(true, SeqCst);
            let (writes, faults, faulting) = writer.join().unwrap();
            let [blocks, calls, began, left] = [0, 1, 2, 3].map(|i| RESTED[i].load(SeqCst));
            println!(
                "writes {pause} us apart, rest {rest} us or {most} checks: {checks} checks took {:?} ({:?} each), reported {reported} blocks written (each a copy); \
                 {blocks} blocks made read-only in {calls} calls; {began} rests begun, {left} resting blocks left open; \
                 {writes} writes, {faults} faulted ({:?} each)",
                checking,
                checking / checks.max(1),
                faulting / (faults.max(1) as u32)
            );
            REST_HERE.set(0);
            title_memory_remapped(at, BLOCKS * BLOCK);
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ }) };
        }
        stop.store(true, SeqCst);
        for s in spinners {
            s.join().unwrap();
        }
    }
}

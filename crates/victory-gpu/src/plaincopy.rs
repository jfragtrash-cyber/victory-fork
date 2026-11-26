//! Copies of title memory made in place, by this process, with a way back
//! when a page is not there (`VICTORY_COPY_PLAIN=1`; off by default).
//!
//! Every copy of title memory into a host buffer goes through the kernel
//! (`ReadProcessMemory` on our own process, `crate::host::copy_mapped`):
//! the title unmaps and remaps its memory while we read it, a read in
//! place of a page that is gone is an access violation, and the kernel's
//! copy fails instead. It costs what a system call that locks and probes
//! every page costs: run 225's profile has the submitting thread 35 ms a
//! frame inside `ZwReadVirtualMemory` for about 100 MB (3 GB a second),
//! where the processor's own copy moves that much in a third of the time.
//!
//! Here the copy is one `rep movsb` in a function of ours that uses no
//! stack. An exception handler, first in the line, looks at every access
//! violation: one that is a READ and happened at exactly that instruction
//! continues at the function's other exit, which returns 0, and the
//! caller falls back to the kernel's copy as before (it says which page
//! is gone and fills in zeros). A write that faults is not ours to take:
//! the destination is host memory, which is always there; such a fault
//! goes on to the handlers after this one.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_copy_guarded",
    ".globl victory_copy_guarded_at",
    ".globl victory_copy_guarded_fault",
    "victory_copy_guarded:",
    // rcx: to, rdx: from, r8: bytes. rsi and rdi are the caller's: kept in
    // r9 and r10, which are ours to use, so nothing is on the stack.
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_copy_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_copy_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
    // The same copy into title memory ([`copy_to_title`]): a fault at its
    // instruction, read or write, ends it with 0.
    ".globl victory_copy_to_title",
    ".globl victory_copy_to_title_at",
    ".globl victory_copy_to_title_fault",
    "victory_copy_to_title:",
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_copy_to_title_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_copy_to_title_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_copy_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_guarded_at: u8;
    static victory_copy_guarded_fault: u8;
    fn victory_copy_to_title(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_to_title_at: u8;
    static victory_copy_to_title_fault: u8;
}

/// Whether copies of title memory into host memory are made in place
/// (`VICTORY_COPY_PLAIN=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(target_arch = "x86_64") && std::env::var_os("VICTORY_COPY_PLAIN").is_some_and(|v| v == "1");
        if on {
            install();
            tracing::info!(
                "copies of title memory into host buffers are made in place, with the kernel's copy behind them for a \
                 page that is not there (VICTORY_COPY_PLAIN=1)"
            );
        }
        on
    })
}

/// Copies made in place, their bytes, the copies that met a page that was
/// not there (and went to the kernel's copy), and the copies not tried in
/// place because their source faulted lately ([`RECENT`]) or they were
/// under [`LEAST`] bytes; then writes into title memory made in place
/// ([`copy_to_title`]), their bytes, and those that met a page not there or
/// not writable; since the last line ([`count`]).
static COUNT: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

thread_local! {
    /// This thread's part of [`COUNT`] not added yet, and the copies since
    /// it was last added.
    static TALLY: ([std::cell::Cell<u64>; 8], std::cell::Cell<u32>) = const {
        ([const { std::cell::Cell::new(0) }; 8], std::cell::Cell::new(0))
    };
}

/// Adds `n` to [`COUNT`]'s place `i`, by way of this thread's tally: added
/// to the shared counters every 256 copies. Every copy of every thread
/// added to the same few counters (the texture threads copy too): in run
/// P that was 1.1 ms a frame of the submitting thread in the adds alone,
/// for some 16,000 small copies a frame.
fn count(i: usize, n: u64) {
    TALLY.with(|(tally, copies)| {
        tally[i].set(tally[i].get() + n);
        if i != 1 && i != 6 {
            copies.set(copies.get() + 1);
        }
        if copies.get() >= 256 {
            copies.set(0);
            for (k, t) in tally.iter().enumerate() {
                let v = t.replace(0);
                if v != 0 {
                    COUNT[k].fetch_add(v, Relaxed);
                }
            }
        }
    });
}

/// The fewest bytes copied in place. A read of a byte or a word is how the
/// layer ASKS whether a page is there (a texture's first page, a record):
/// it is expected to fail, and a failed kernel read costs a microsecond
/// where a fault and its handler cost tens (run 230: 1,000 to 1,300 faults
/// a frame, nearly all of them such askings, took back most of what the
/// copies in place had won).
const LEAST: usize = 256;

/// The 64 KiB blocks of title memory a copy in place faulted in lately:
/// their copies go straight to the kernel's, which reads round the hole.
/// By the block's number, in the place its low bits name (a later block
/// with the same low bits takes the place). Forgotten every two seconds:
/// the title maps the memory in the end.
static RECENT: [std::sync::atomic::AtomicUsize; 1024] = [const { std::sync::atomic::AtomicUsize::new(0) }; 1024];
/// When [`RECENT`] was last emptied (milliseconds of [`now_ms`]).
static EMPTIED: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// The numbers of the first and last 64 KiB block of `[from, from + len)`,
/// never 0 (0 is an empty place).
fn blocks(from: usize, len: usize) -> (usize, usize) {
    ((from >> 16) | 1 << 63, ((from + len.max(1) - 1) >> 16) | 1 << 63)
}

/// Whether a copy from there faulted lately.
fn faulted_lately(from: usize, len: usize) -> bool {
    let now = now_ms();
    let emptied = EMPTIED.load(Relaxed);
    if now.saturating_sub(emptied) >= 2_000 && EMPTIED.compare_exchange(emptied, now, Relaxed, Relaxed).is_ok() {
        for place in &RECENT {
            place.store(0, Relaxed);
        }
        return false;
    }
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].load(Relaxed) == first || RECENT[last & 1023].load(Relaxed) == last
}

/// A copy from there faulted.
fn note_fault(from: usize, len: usize) {
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].store(first, Relaxed);
    RECENT[last & 1023].store(last, Relaxed);
}

/// The handler that turns a read fault inside [`guarded`] into its return
/// of false; put first in the line, once.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it will continue with.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        // The copy into title memory: any fault there, read or write (a page
        // the title unmapped), ends it with 0 ([`copy_to_title`]).
        if context.Rip == (&raw const victory_copy_to_title_at) as u64 {
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR
                || r.ExceptionCode == EXCEPTION_GUARD_PAGE
            {
                context.Rip = (&raw const victory_copy_to_title_fault) as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // Only at the copy's own instruction.
        if context.Rip != (&raw const victory_copy_guarded_at) as u64 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const READ: usize = 0;
        let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
        let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
        if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
            context.Rip = (&raw const victory_copy_guarded_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that reads the record it is given and nothing else.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn install() {}

/// Copies `len` bytes from `from` to `to` in place; false if a page of the
/// source was not there (then some of `to` may have been written).
///
/// `to` must be writable memory of `len` bytes that is ours (host memory);
/// `from` may be anything.
#[cfg(target_arch = "x86_64")]
fn guarded(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the destination is the caller's to write; a source page that
    // is not there ends the copy with 0 instead of a fault ([`install`]).
    unsafe { victory_copy_guarded(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// [`guarded`], whatever [`on`] says: for tests elsewhere of what faults
/// in it.
#[cfg(test)]
pub(crate) fn guarded_copy(to: *mut u8, from: *const u8, len: usize) -> bool {
    guarded(to, from, len)
}

/// With [`on`]: copies title memory at `from` into host memory at `to`;
/// false if it is off or a source page was not there (the caller then
/// copies through the kernel, as before). Under [`LEAST`] bytes it is left
/// to the kernel: such reads are mostly askings whether a page is there.
pub(crate) fn copy(to: *mut u8, from: *const u8, len: usize) -> bool {
    copy_from(to, from, len, LEAST)
}

/// [`copy`] for a copy of title memory that is expected to be there
/// whatever its length (`crate::host::copy_mapped`: a list's constants
/// filled into its upload memory, a refresh): in place from one byte on
/// (`VICTORY_COPY_PLAIN_SMALL`; on by default, `0` leaves those under
/// [`LEAST`] bytes to the kernel, as before).
///
/// Run P's submissions filled some 16,000 constant buffers a frame (8.5
/// MB, 535 bytes on average), most of them under 256 bytes and so read
/// through the kernel: 5.7 ms a frame of the submitting thread in
/// `ZwReadVirtualMemory`, about half a microsecond each. A source that is
/// not there faults once and is then left to the kernel for a while
/// ([`RECENT`]), as for longer copies.
pub(crate) fn copy_any(to: *mut u8, from: *const u8, len: usize) -> bool {
    static SMALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let small = *SMALL.get_or_init(|| std::env::var_os("VICTORY_COPY_PLAIN_SMALL").is_none_or(|v| v != "0"));
    copy_from(to, from, len, if small { 1 } else { LEAST })
}

/// [`copy`], in place from `least` bytes on.
fn copy_from(to: *mut u8, from: *const u8, len: usize, least: usize) -> bool {
    if !on() {
        return false;
    }
    if len < least {
        count(4, 1);
        return false;
    }
    if faulted_lately(from as usize, len) {
        count(3, 1);
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        // The bytes first: they are added with the copy that ends a batch.
        count(1, len as u64);
        count(0, 1);
    } else {
        count(2, 1);
        note_fault(from as usize, len);
    }
    whole
}

/// Whether bytes put into title memory (what shaders wrote, landed by the
/// relay: `crate::host::write_mapped`) are copied in place
/// (`VICTORY_WRITE_PLAIN`; on with [`on`] unless `0`).
///
/// Every piece went through the kernel (`NtWriteVirtualMemory`), which
/// leaves a page that is not there alone: run AF's copy-back wrote some
/// 54 MB a frame in about 11,700 stretches of 4 KiB, one system call each,
/// and the relay's landing stayed at ~52 ms a frame with its merge on four
/// threads.
fn write_plain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_PLAIN").is_none_or(|v| v != "0"))
}

/// With [`on`] and [`write_plain`]: copies host memory at `from` into title
/// memory at `to` in place; false if off or a page of the destination was
/// not there or not writable (then some of `to` may have been written, and
/// the caller writes through the kernel, which leaves such pages alone).
/// The caller opens the destination's guards first, as for the kernel's
/// write (`crate::watch::writing`).
pub(crate) fn copy_to_title(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() || !write_plain() || len == 0 {
        return false;
    }
    let whole = to_title(to, from, len);
    if whole {
        count(6, len as u64);
        count(5, 1);
    } else {
        count(7, 1);
    }
    whole
}

/// [`copy_to_title`]'s copy, whatever the switches.
#[cfg(target_arch = "x86_64")]
fn to_title(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the source is host memory of `len` bytes; a destination page
    // that is not there, or not writable, ends the copy with 0 instead of a
    // fault ([`install`]).
    unsafe { victory_copy_to_title(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn to_title(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// With the frame line: what [`copy`] did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [copies, bytes, gone, lately, small, writes, written, refused] =
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "copies of title memory made in place per frame (VICTORY_COPY_PLAIN=1): {copies:.0}, {:.1} MB; {gone:.1} met a page \
         that was not there and went through the kernel's copy; left to the kernel's copy untried: {lately:.0} whose source \
         faulted in the last two seconds, {small:.0} of under {LEAST} bytes; writes into title memory in place \
         (VICTORY_WRITE_PLAIN): {writes:.0}, {:.1} MB, {refused:.1} met a page not there or not writable and went to the \
         kernel",
        bytes / 1e6,
        written / 1e6
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    #[test]
    fn memory_that_is_there_is_copied() {
        let from: Vec<u8> = (0..100_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut to = vec![0u8; from.len()];
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), from.len()));
        assert_eq!(to, from);
        // Nothing to copy is a copy.
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), 0));
    }

    /// Three pages reserved, the first committed: a copy that stays in the
    /// first page is whole; one that runs into the second ends with false,
    /// the bytes before the hole copied, and the thread goes on.
    #[test]
    fn a_page_that_is_not_there_ends_the_copy_without_a_fault() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
        let mut to = vec![0u8; 2 * PAGE];
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        to.fill(0);
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        assert!(to[..PAGE / 2].iter().all(|&b| b == 0xAB), "the bytes before the hole");
        assert!(to[PAGE / 2..].iter().all(|&b| b == 0), "nothing after it");
        // A source that is not there at all.
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));
        // Committed but closed to reads.
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(2 * PAGE), 16));
        // And the registers the caller owns came back: this still works.
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// A copy into memory that is there is whole; one that meets a page not
    /// there, or read-only, ends with false, the bytes before it written,
    /// and the thread goes on.
    #[test]
    fn a_destination_that_is_not_there_or_read_only_ends_the_copy() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first and third pages of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let from = vec![0xCDu8; 2 * PAGE];
        assert!(to_title(base, from.as_ptr(), PAGE));
        // SAFETY: committed and readable.
        assert!(unsafe { std::slice::from_raw_parts(base, PAGE) }.iter().all(|&b| b == 0xCD));
        // Into the hole: the half before it written.
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0);
        assert!(!to_title(base.wrapping_add(PAGE / 2), from.as_ptr(), PAGE));
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(base.wrapping_add(PAGE / 2), PAGE / 2) }.iter().all(|&b| b == 0xCD));
        // A read-only page (a guard not opened).
        let mut old = 0;
        // SAFETY: the third page, ours.
        assert!(unsafe { VirtualProtect(base.wrapping_add(2 * PAGE).cast(), PAGE, PAGE_READONLY, &mut old) } != 0);
        assert!(!to_title(base.wrapping_add(2 * PAGE), from.as_ptr(), 16));
        // SAFETY: committed and readable.
        assert!(unsafe { std::slice::from_raw_parts(base.wrapping_add(2 * PAGE), PAGE) }.iter().all(|&b| b == 0));
        // The registers the caller owns came back: this still works.
        assert!(to_title(base, from.as_ptr(), PAGE));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// A source that faulted is remembered by its 64 KiB blocks, and only
    /// those.
    #[test]
    fn a_source_that_faulted_is_remembered_by_its_blocks() {
        // (Addresses no other test uses; the table is shared.)
        let at = 0x7654_3210_0000usize;
        assert!(!faulted_lately(at, 4096));
        note_fault(at + 100, 4096);
        assert!(faulted_lately(at, 16));
        assert!(faulted_lately(at + 0xffff, 1));
        // A copy that ends in the block, and one that begins in it.
        assert!(faulted_lately(at - 4096, 8192));
        // The blocks beside it are not remembered.
        assert!(!faulted_lately(at + 0x10000, 4096));
        assert!(!faulted_lately(at - 0x10000, 4096));
        assert_eq!(blocks(0, 0), (1 << 63, 1 << 63));
    }

    /// A thread's counts reach the shared counters every 256 copies, the
    /// bytes with them.
    #[test]
    fn counts_reach_the_shared_counters_in_batches() {
        let [copies, bytes] = [0, 1].map(|i| COUNT[i].load(Relaxed));
        std::thread::spawn(|| {
            for _ in 0..300 {
                count(1, 10);
                count(0, 1);
            }
        })
        .join()
        .unwrap();
        // (Nothing else counts here: copies in place are off in tests.)
        assert_eq!(COUNT[0].load(Relaxed) - copies, 256);
        assert_eq!(COUNT[1].load(Relaxed) - bytes, 2560);
    }

    /// Many threads at once, half of them faulting: each gets its own answer.
    #[test]
    fn threads_fault_and_copy_side_by_side() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(base, 0);
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut to = vec![0u8; PAGE];
                    for _ in 0..2_000 {
                        let from = (base + if t % 2 == 0 { 0 } else { PAGE }) as *const u8;
                        assert_eq!(guarded(to.as_mut_ptr(), from, PAGE), t % 2 == 0);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}

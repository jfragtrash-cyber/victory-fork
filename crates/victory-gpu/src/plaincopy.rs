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
    // The copy of [`copy_streamed`]: 32 bytes a load, 128 a turn, the
    // turns' stores past the caches (`vmovntdq`, to the destination's
    // 32-byte lines). rcx: to, rdx: from, r8: bytes, at least 128. Only
    // registers the caller does not keep (rax, rcx, rdx, r8 to r11, ymm0 to
    // ymm4) and no stack: a fault anywhere from `victory_copy_streamed_at`
    // (its first instruction) to `victory_copy_streamed_end` may continue at
    // `victory_copy_streamed_fault`, which returns 0.
    // First the head (the first 32 bytes, stored last) and the destination
    // moved up to its next 32-byte line; then turns of 128 while more than
    // 128 are left; then the last 128 bytes, counted back from the end
    // (they may cover bytes a turn copied). The fence puts the stores past
    // the caches before whatever the caller stores next (the word that the
    // copy is done, which another thread or the card then reads).
    ".globl victory_copy_streamed",
    ".globl victory_copy_streamed_at",
    ".globl victory_copy_streamed_end",
    ".globl victory_copy_streamed_fault",
    ".p2align 5",
    "victory_copy_streamed:",
    "victory_copy_streamed_at:",
    "    lea r9, [rcx + r8]",
    "    lea rax, [rdx + r8]",
    "    vmovdqu ymm4, ymmword ptr [rdx]",
    "    mov r10, rcx",
    "    mov r11, rcx",
    "    neg r11",
    "    and r11, 31",
    "    add rcx, r11",
    "    add rdx, r11",
    "    sub r8, r11",
    "    sub rdx, rcx",
    "    cmp r8, 128",
    "    jbe 3f",
    ".p2align 5",
    "2:",
    "    vmovdqu ymm0, ymmword ptr [rcx + rdx]",
    "    vmovdqu ymm1, ymmword ptr [rcx + rdx + 32]",
    "    vmovdqu ymm2, ymmword ptr [rcx + rdx + 64]",
    "    vmovdqu ymm3, ymmword ptr [rcx + rdx + 96]",
    "    vmovntdq ymmword ptr [rcx], ymm0",
    "    vmovntdq ymmword ptr [rcx + 32], ymm1",
    "    vmovntdq ymmword ptr [rcx + 64], ymm2",
    "    vmovntdq ymmword ptr [rcx + 96], ymm3",
    "    sub rcx, -128",
    "    add r8, -128",
    "    cmp r8, 128",
    "    ja 2b",
    "3:",
    "    vmovdqu ymm0, ymmword ptr [rax - 128]",
    "    vmovdqu ymm1, ymmword ptr [rax - 96]",
    "    vmovdqu ymm2, ymmword ptr [rax - 64]",
    "    vmovdqu ymm3, ymmword ptr [rax - 32]",
    "    vmovdqu ymmword ptr [r9 - 128], ymm0",
    "    vmovdqu ymmword ptr [r9 - 96], ymm1",
    "    vmovdqu ymmword ptr [r9 - 64], ymm2",
    "    vmovdqu ymmword ptr [r9 - 32], ymm3",
    "    vmovdqu ymmword ptr [r10], ymm4",
    "    sfence",
    "    vzeroupper",
    "    mov eax, 1",
    "    ret",
    "victory_copy_streamed_end:",
    "victory_copy_streamed_fault:",
    "    sfence",
    "    vzeroupper",
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
    fn victory_copy_streamed(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_streamed_at: u8;
    static victory_copy_streamed_end: u8;
    static victory_copy_streamed_fault: u8;
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
/// not writable; then the bytes copied with the stores past the caches
/// ([`copy_streamed`], [`copy_own_streamed`]); since the last line
/// ([`count`]).
static COUNT: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

thread_local! {
    /// This thread's part of [`COUNT`] not added yet, and the copies since
    /// it was last added.
    static TALLY: ([std::cell::Cell<u64>; 9], std::cell::Cell<u32>) = const {
        ([const { std::cell::Cell::new(0) }; 9], std::cell::Cell::new(0))
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
        if !matches!(i, 1 | 6 | 8) {
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

/// Milliseconds since the first call, from the system's tick count (a read
/// of memory the kernel keeps, some 16 ms fine; [`RECENT`] forgets every
/// two seconds). It was the performance counter, asked with every copy in
/// place: 0.45 ms a frame of the submitting thread in run BA's gameplay
/// (`faulted_lately`), as `crate::host::moment` once was.
fn now_ms() -> u64 {
    static START: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    // SAFETY: a plain query.
    let now = unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() };
    now.saturating_sub(*START.get_or_init(|| now))
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

/// The handler that turns a read fault inside [`guarded`] or [`streamed`]
/// into its return of false; put first in the line, once.
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
        // Only at the copy's own instruction, or anywhere in the streamed
        // copy (its loads read the source; its stores, whose faults are not
        // ours, say "write").
        let rip = context.Rip;
        let streamed =
            (&raw const victory_copy_streamed_at) as u64 <= rip && rip < (&raw const victory_copy_streamed_end) as u64;
        if rip != (&raw const victory_copy_guarded_at) as u64 && !streamed {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const READ: usize = 0;
        let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
        let gone = (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
        if gone || r.ExceptionCode == EXCEPTION_GUARD_PAGE {
            context.Rip = if streamed {
                (&raw const victory_copy_streamed_fault) as u64
            } else {
                (&raw const victory_copy_guarded_fault) as u64
            };
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

/// [`guarded`] with the stores past the caches; `len` at least 128.
#[cfg(target_arch = "x86_64")]
fn streamed(to: *mut u8, from: *const u8, len: usize) -> bool {
    debug_assert!(len >= 128);
    install();
    // SAFETY: as in `guarded`. The copy finds the destination's 32-byte
    // lines itself; AVX is there ([`stream`] asked).
    unsafe { victory_copy_streamed(to, from, len) != 0 }
}

#[cfg(not(target_arch = "x86_64"))]
fn streamed(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// The fewest bytes a copy stores past the caches ([`stream`]): the fence
/// at its end waits for its last stores to reach memory, some 0.3-0.5 us.
/// From memory far larger than the caches (`copy_speeds`), pieces of 16 KiB
/// went from 13.6 to 20.5 GB/s; of 4 KiB, from 13.5 to 14.2.
const STREAM_LEAST: usize = 16 << 10;

/// Whether copies into host memory that this processor does not read soon
/// store past the caches ([`copy_streamed`], [`copy_own_streamed`]):
/// `VICTORY_COPY_STREAM=1`, `alt` twenty seconds on and twenty off (each
/// change logged); off by default. Needs AVX.
///
/// A copy's ordinary stores first read every line they write (the
/// processor owns a line before it changes it), and leave it in the caches:
/// a host copy's refresh moves its bytes three times through memory and
/// pushes 70 MB a frame (run BL) through caches of 32 MB, though only the
/// card reads them, frames later. Run BA's kept threads spent 28.7 ms a
/// frame in `rep movsb` on those refreshes, about 2.4 GB a second each,
/// and the submitting thread waited 5.6 ms a frame for kept threads.
/// Stores past the caches neither read the line nor keep it: on this
/// machine (5950X, `copy_speeds`) copies of 64 KiB pieces between buffers
/// far larger than the caches went from 12.4 to 20.7 GB/s on one thread,
/// and from 13.3 to 21.3 GB/s on seven at once. Into a buffer that is read
/// again at once (a texture's read, a page to merge) it is the other way
/// (34.4 against 20.4 GB/s): those copies stay as they were.
fn stream() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_COPY_STREAM").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        #[cfg(target_arch = "x86_64")]
        let avx = std::arch::is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;
        if mode != 0 && !avx {
            tracing::warn!("VICTORY_COPY_STREAM needs AVX, which this processor lacks: copies as before");
            return 0;
        }
        if mode != 0 {
            install();
        }
        mode
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            let on = (now_ms() / 20_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            // (Read first: every copy of eight threads swapping one word
            // would pass its line from core to core.)
            if LAST.load(Relaxed) != u8::from(on) && LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                tracing::info!(
                    "host copies' refreshes and snapshots store past the caches (VICTORY_COPY_STREAM=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
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

/// [`copy_any`] for a copy into host memory that this processor does not
/// read soon: a host copy's refresh (the card reads it; the next refresh
/// comes frames later). With [`stream`], from [`STREAM_LEAST`] bytes on, the
/// stores go past the caches.
pub(crate) fn copy_streamed(to: *mut u8, from: *const u8, len: usize) -> bool {
    if len < STREAM_LEAST || !on() || !stream() {
        return copy_any(to, from, len);
    }
    copy_from_with(to, from, len, 1, true)
}

/// Copies `len` bytes of host memory from `from` to `to`, as
/// `std::ptr::copy_nonoverlapping` does; with [`stream`], from
/// [`STREAM_LEAST`] bytes on, the stores go past the caches. For copies
/// nothing on this processor reads soon: the snapshots of ranges shaders
/// write, read when their flight is done.
///
/// # Safety
/// As for `std::ptr::copy_nonoverlapping`.
pub(crate) unsafe fn copy_own_streamed(to: *mut u8, from: *const u8, len: usize) {
    if len >= STREAM_LEAST && stream() && streamed(to, from, len) {
        count(8, len as u64);
        return;
    }
    // (Memory of ours that is not there faults in the streamed copy, which
    // gives up, and then here, as it always did.)
    // SAFETY: the caller's.
    unsafe { std::ptr::copy_nonoverlapping(from, to, len) };
}

/// [`copy`], in place from `least` bytes on.
fn copy_from(to: *mut u8, from: *const u8, len: usize, least: usize) -> bool {
    copy_from_with(to, from, len, least, false)
}

/// [`copy_from`], the stores past the caches if `stream` (then `len` is at
/// least 128).
fn copy_from_with(to: *mut u8, from: *const u8, len: usize, least: usize, stream: bool) -> bool {
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
    let whole = if stream { streamed(to, from, len) } else { guarded(to, from, len) };
    if whole {
        // The bytes first: they are added with the copy that ends a batch.
        count(1, len as u64);
        if stream {
            count(8, len as u64);
        }
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
    let [copies, bytes, gone, lately, small, writes, written, refused, streamed] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "copies of title memory made in place per frame (VICTORY_COPY_PLAIN=1): {copies:.0}, {:.1} MB; {gone:.1} met a page \
         that was not there and went through the kernel's copy; left to the kernel's copy untried: {lately:.0} whose source \
         faulted in the last two seconds, {small:.0} of under {LEAST} bytes; writes into title memory in place \
         (VICTORY_WRITE_PLAIN): {writes:.0}, {:.1} MB, {refused:.1} met a page not there or not writable and went to the \
         kernel; {:.1} MB stored past the caches, snapshots with them (VICTORY_COPY_STREAM)",
        bytes / 1e6,
        written / 1e6,
        streamed / 1e6
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOMBINE, VirtualAlloc, VirtualFree,
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

    /// The streamed copy copies every byte, whatever the length and the
    /// lines either side starts on, and nothing beside them.
    #[test]
    fn the_streamed_copy_copies_every_byte_and_nothing_beside() {
        if !std::arch::is_x86_feature_detected!("avx") {
            return;
        }
        let from: Vec<u8> = (0..300_000u32).map(|i| (i * 131 + i / 251) as u8).collect();
        let mut to = vec![0u8; from.len() + 128];
        let lens = (128..700).chain([1000, 4095, 4096, 4097, 16 << 10, 65_536 + 33, 200_000]);
        for len in lens {
            for (a, b) in [(0, 0), (1, 0), (0, 1), (7, 31), (31, 7), (32, 32), (63, 17)] {
                let to = &mut to[..len + 128];
                to.fill(0xEE);
                assert!(streamed(to.as_mut_ptr().wrapping_add(b), from.as_ptr().wrapping_add(a), len));
                assert_eq!(to[b..b + len], from[a..a + len], "{len} bytes, from +{a} to +{b}");
                assert!(to[..b].iter().chain(&to[b + len..]).all(|&x| x == 0xEE), "{len} bytes: beside");
            }
        }
    }

    /// The streamed copy that meets a page that is not there, or closed to
    /// reads, ends with false and the thread goes on, its registers whole.
    #[test]
    fn a_streamed_copy_that_meets_a_hole_ends_without_a_fault() {
        if !std::arch::is_x86_feature_detected!("avx") {
            return;
        }
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
        let mut to = vec![0u8; 2 * PAGE];
        assert!(streamed(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // Into the hole from inside the page, in the turns and in the last
        // 128 bytes; a source not there at all; one closed to reads.
        assert!(!streamed(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        assert!(!streamed(to.as_mut_ptr(), base.wrapping_add(PAGE - 200), 300));
        assert!(!streamed(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        assert!(!streamed(to.as_mut_ptr(), base.wrapping_add(2 * PAGE), 128));
        // The registers the caller owns came back: this still works.
        to.fill(0);
        assert!(streamed(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // And the copy with `rep movsb` beside it.
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// Times the copies: `rep movsb` (the guarded copy as it was), the
    /// runtime's `memcpy`, and the streamed copy, on memory far larger
    /// than the caches (source and destination at the same place in their
    /// 64 KiB lines, as a title page and its host copy are), on one thread
    /// and on seven at once (the kept threads); then from such memory into
    /// one reused buffer and into write-combined memory, and small copies
    /// in the caches. 5950X, GB/s (rep movsb / memcpy / streamed): pieces
    /// of 4 KiB 13.5 / 13.7 / 14.2, 16 KiB 13.6 / 13.2 / 20.5, 64 KiB 12.4 /
    /// 13.1 / 20.7, 256 KiB 13.7 / 13.8 / 21.9; seven threads, 64 KiB, 13.3
    /// / 13.6 / 21.3 in all; into one reused buffer, 64 KiB, 34.4 / 33.6 /
    /// 20.4. (Other runs, with builds beside them: rep movsb 9.5-11.8, the
    /// streamed copy 1.4-1.7 times that.)
    /// `cargo test --profile diag -p victory-gpu --lib copy_speeds -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn copy_speeds() {
        type Copy = unsafe extern "C" fn(*mut u8, *const u8, usize) -> u32;
        unsafe extern "C" fn crt(to: *mut u8, from: *const u8, len: usize) -> u32 {
            // SAFETY: the bench's own buffers.
            unsafe { std::ptr::copy_nonoverlapping(from, to, len) };
            1
        }
        let ways: [(&str, Copy); 3] =
            [("rep movsb", victory_copy_guarded), ("memcpy", crt), ("streamed", victory_copy_streamed)];
        const SPAN: usize = 1 << 30;
        let alloc = |n: usize| {
            // SAFETY: memory of the bench's own, touched whole before timing.
            let p = unsafe { VirtualAlloc(std::ptr::null(), n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
            assert!(!p.is_null());
            // SAFETY: just committed.
            unsafe { std::ptr::write_bytes(p, 0x5A, n) };
            p as usize
        };
        let (src, dst) = (alloc(SPAN), alloc(SPAN + (64 << 10)));
        for &(threads, piece, apart) in &[
            (1usize, 4usize << 10, 0usize),
            (1, 16 << 10, 0),
            (1, 64 << 10, 0),
            (1, 64 << 10, 2048),
            (1, 256 << 10, 0),
            (1, 1 << 20, 0),
            (7, 64 << 10, 0),
            (7, 256 << 10, 0),
        ] {
            for &(name, way) in &ways {
                let slice = SPAN / threads;
                let started = std::time::Instant::now();
                let total: usize = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            s.spawn(move || {
                                let (from, to) = (src + t * slice, dst + t * slice + apart);
                                let mut copied = 0;
                                for _ in 0..2 {
                                    let mut at = 0;
                                    while at + piece <= slice {
                                        // SAFETY: inside the bench's buffers.
                                        unsafe { way((to + at) as *mut u8, (from + at) as *const u8, piece) };
                                        at += piece;
                                        copied += piece;
                                    }
                                }
                                copied
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).sum()
                });
                let s = started.elapsed().as_secs_f64();
                println!(
                    "{threads} thread(s), pieces of {:>4} KiB{}: {name:>10} {:6.2} GB/s ({:.1} ms per 70 MB)",
                    piece >> 10,
                    if apart != 0 { ", 2 KiB apart" } else { "" },
                    total as f64 / s / 1e9,
                    70e6 / (total as f64 / s) * 1e3
                );
            }
        }
        // A cold source into one reused buffer (a texture read, a page to
        // merge), and into write-combined memory (the upload memory fills go
        // to), one thread.
        let scratch = alloc(1 << 20);
        // SAFETY: memory of the bench's own, touched whole before timing.
        let combined = unsafe {
            VirtualAlloc(std::ptr::null(), 64 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE)
        } as usize;
        assert_ne!(combined, 0);
        // SAFETY: just committed.
        unsafe { std::ptr::write_bytes(combined as *mut u8, 0, 64 << 20) };
        for &(piece, into, what) in &[
            (64usize << 10, scratch, "into one reused buffer"),
            (1 << 20, scratch, "into one reused buffer"),
            (535, combined, "into write-combined memory"),
            (4 << 10, combined, "into write-combined memory"),
            (64 << 10, combined, "into write-combined memory"),
        ] {
            for &(name, way) in &ways {
                let started = std::time::Instant::now();
                let (mut at, mut copied, mut to) = (0, 0, 0);
                while copied < (256 << 20) {
                    // SAFETY: inside the bench's buffers.
                    unsafe { way((into + to) as *mut u8, (src + at) as *const u8, piece) };
                    at = (at + piece.next_multiple_of(64)) % (SPAN - piece);
                    if into == combined {
                        to = (to + piece.next_multiple_of(64)) % ((64 << 20) - piece);
                    }
                    copied += piece;
                }
                let s = started.elapsed().as_secs_f64();
                println!(
                    "pieces of {:>7} bytes {what}: {name:>10} {:6.2} GB/s ({:.1} ns a piece)",
                    piece,
                    copied as f64 / s / 1e9,
                    s * 1e9 / (copied / piece) as f64
                );
            }
        }
        // SAFETY: the bench's own.
        unsafe {
            VirtualFree(src as *mut _, 0, MEM_RELEASE);
            VirtualFree(dst as *mut _, 0, MEM_RELEASE);
            VirtualFree(scratch as *mut _, 0, MEM_RELEASE);
            VirtualFree(combined as *mut _, 0, MEM_RELEASE);
        }
        // Small copies in the caches.
        let from = vec![0x33u8; 64 << 10];
        let mut to = vec![0u8; 64 << 10];
        for &len in &[128usize, 256, 535, 1024, 4096, 16384] {
            for &(name, way) in &ways {
                let rounds = 2_000_000 / (len / 128).max(1);
                let started = std::time::Instant::now();
                for i in 0..rounds {
                    let at = (i * 4160) % ((64 << 10) - len);
                    // SAFETY: inside the two buffers.
                    unsafe { way(to.as_mut_ptr().add(at), from.as_ptr().add(at), len) };
                }
                let ns = started.elapsed().as_nanos() as f64 / rounds as f64;
                println!("in the caches, {len:>5} bytes: {name:>10} {ns:7.1} ns a copy");
            }
        }
        std::hint::black_box(&to);
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

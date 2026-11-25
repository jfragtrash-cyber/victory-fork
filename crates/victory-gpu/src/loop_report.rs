//! Diagnostics (`VICTORY_SHADER_LOOP_REPORT`): the loop report translated
//! shaders write when the iteration limit stops a lane (layout at
//! [`victory_shader::emit::loop_report`]). It sits in system memory, so it
//! still reads after the device is removed; logged by loop, the lanes
//! stopped in each.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use victory_shader::emit::{LOOP_REPORT_BYTES, LOOP_REPORT_SLOTS};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// The buffer and where it is mapped.
static REPORT: OnceLock<Option<(ID3D12Resource, usize)>> = OnceLock::new();

/// The report's GPU address; the buffer is made, zeroed, on first use.
pub(crate) fn gpu() -> Option<u64> {
    let (buffer, _) = REPORT
        .get_or_init(|| {
            let (buffer, data) = crate::host::host()?.shared_buffer(LOOP_REPORT_BYTES)?;
            // SAFETY: the mapping covers the buffer.
            unsafe { std::ptr::write_bytes(data, 0, LOOP_REPORT_BYTES as usize) };
            Some((buffer, data as usize))
        })
        .as_ref()?;
    // SAFETY: a plain query.
    Some(unsafe { buffer.GetGPUVirtualAddress() })
}

/// After a submission: when lanes were stopped since the last log, logs
/// the loops that stopped them, most lanes first; at most every 2 s,
/// unless the device was `removed`.
pub(crate) fn log(removed: bool) {
    let Some(Some((_, data))) = REPORT.get() else {
        return;
    };
    let words = *data as *const u32;
    // SAFETY: inside the mapping; shaders may still write it.
    let read = |i: usize| unsafe { std::ptr::read_volatile(words.add(i)) };
    static LAST: Mutex<(u32, Option<Instant>)> = Mutex::new((0, None));
    let total = read(0);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let recent = last.1.is_some_and(|t| t.elapsed() < Duration::from_secs(2));
        if total == last.0 || recent && !removed {
            return;
        }
        *last = (total, Some(Instant::now()));
    }
    // (Lanes, tag, console offset of the loop header.)
    let mut loops: Vec<(u32, u32, u32)> = (0..LOOP_REPORT_SLOTS as usize)
        .filter_map(|s| {
            let at = 4 + 4 * s;
            let tag = read(at);
            (tag != 0).then(|| (read(at + 2), tag, read(at + 1).wrapping_sub(1)))
        })
        .collect();
    loops.sort_unstable_by(|a, b| b.cmp(a));
    let names = crate::pipeline::by_report_tag();
    tracing::warn!(
        lanes = total,
        lost = read(1),
        loops = loops.len(),
        "translated loops stopped at the iteration limit"
    );
    let dump = std::env::var_os("VICTORY_SHADER_LOOP_DUMP").map(std::path::PathBuf::from);
    for &(lanes, tag, offset) in loops.iter().take(30) {
        let Some(s) = names.get(&tag) else {
            tracing::warn!(lanes, offset = format_args!("{offset:#x}"), "loop stopped at the limit in unknown");
            continue;
        };
        tracing::warn!(
            lanes,
            offset = format_args!("{offset:#x}"),
            "loop stopped at the limit in {} {}",
            s.stage,
            s.hash
        );
        // `VICTORY_SHADER_LOOP_DUMP=<dir>`: the console container, once.
        if let Some(dir) = &dump {
            let path = dir.join(format!("{}-{}.dxbc", s.stage, s.hash));
            if !path.exists()
                && let Err(error) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &s.container))
            {
                tracing::warn!(%error, path = %path.display(), "loop report dump failed");
            }
        }
    }
}

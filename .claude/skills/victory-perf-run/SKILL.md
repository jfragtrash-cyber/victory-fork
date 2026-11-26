---
name: victory-perf-run
description: Launch, drive and measure a world run of the title under VIctory (diagnostics build), then read the numbers - frame lines, frame log, sampling profile, captures. Use whenever a change needs measuring in the real title, when comparing two builds or switch sets, or when hunting what a frame is made of. Only one run of the title can happen on this machine at a time.
---

# Measured runs of the title

One title process at a time on this PC. Before launching, make sure no
`victory.exe` is running (`tasklist | grep -i victory.exe`). Agents that only
edit code must not launch runs; the lead or the `perf-run-analyst` does.

## 1. Build

```bash
cd /d/victory-fork   # or the worktree being measured
cargo build --profile diag -p victory --features diagnostics   # target/diag/victory.exe
```

The shader cache key is the translator's source: a build that leaves
`crates/victory-shader` alone keeps its translations. After a translator
change the first world minutes drop draws while shaders translate
(`victory translate --warm all <game dir>` pre-fills the cache offline).

## 2. Restore the saves (every run)

The title autosaves into the user's saves. Put the reference back first:

```bash
cd /d/victory-data/data/7952987A && B=/d/victory-data/saves-backup-20261125-after-capB && \
  { diff -rq saves "$B" >/dev/null 2>&1 || { rm -rf saves && cp -a "$B" saves; }; } && diff -rq saves "$B" && echo restored
```

## 3. Launch (background, with a timeout)

The title's executable is the one `.exe` in `D:/victory-data/game/`
(`ls D:/victory-data/game/*.exe`). Never write its file name into the tree.

Best known switch set (2026-11-26; runs BI/BJ, 75-77 ms a frame at the cell window, captured; the
staging pool stays on; async page mappings are on by default with `VICTORY_ASYNC_SUBMIT=1`):

```bash
BEST="VICTORY_CARD_HASH_WIDE=1 VICTORY_ASYNC_PAGE_MAPPINGS=1 VICTORY_STAGING_POOL_MB=1024 VICTORY_VIEW_KNOWN=1 VICTORY_ASYNC_PRESENT=1 VICTORY_LEVELS_ON_DEMAND=1 VICTORY_ASYNC_SUBMIT=1 \
VICTORY_QUEUE2=1 VICTORY_INDIRECT_WINDOW=1 VICTORY_TEXTURE_NO_WAIT=1 VICTORY_MERGE_GAP=64 \
VICTORY_MERGE_RUNS=4096 VICTORY_COPY_PLAIN=1 VICTORY_JOIN_PARTS=1 \
VICTORY_BACK_OWN_VIEW=1 VICTORY_INDIRECT_GATHER=1 VICTORY_CARD_REHASH_RUNS=1"
EXE=$(ls D:/victory-data/game/*.exe | head -n 1)
mkdir -p /d/victory-data/cap-X && rm -f /d/victory-data/cap-X/*
env NO_COLOR=1 $BEST \
  VICTORY_CAPTURE='D:\victory-data\cap-X' VICTORY_CAPTURE_WORLD=1 \
  VICTORY_KEYS="30:Enter,35:X" \
  VICTORY_LOG="warn,victory_gpu::host=info,victory_gpu::framelog=info,victory_gpu::queue=info,victory_gpu::copystat=info,victory_gpu::capture=info,victory_gpu::draw=info" \
  timeout 480 target/diag/victory.exe run "$EXE" --execute > /d/victory-data/victory-cap-X.log 2>&1
```

Run it with `run_in_background`. Name each run with a new letter (cap-X,
victory-cap-X.log) and say which build and switches it had.

Input (`VICTORY_KEYS=<seconds>:<key>[:<hold>]`, focus or not):
- `30:Enter` presses A at the title screen; `35:X` skips the opening video.
  Without them the run waits at "press A" forever (and the video plays).
  On a busy machine the title screen (13 draws a frame) can come after
  80 s: then repeat the pair, e.g.
  `30:Enter,35:X,50:Enter,55:X,70:Enter,75:X,90:Enter,95:X,110:Enter,115:X,130:Enter,135:X`.
  If the copied exe is named otherwise, look for it by that name
  (`tasklist | grep -i victory`), not `victory.exe`.
- Sticks: W A S D (move), I J K L (camera); Z / C triggers. Holds in
  seconds: `140:W:5` holds W five seconds at 140 s. Gameplay starts about
  280 frames after the world (`first frame of 500 draws`). Walking to the
  yard and canteen means following the guard; a scripted walk that only
  pushes W ends in a wall.

Diagnostics that cost time (leave out when measuring absolute speed):
- `VICTORY_CAPTURE` + `VICTORY_CAPTURE_WORLD=1`: a PNG every 8th world
  frame (~10 ms a frame on the submitting thread). The log line "the first
  frame of 500 draws or more ... frame=N" exists only with capture on.
- `VICTORY_PROFILE=1 VICTORY_PROFILE_OUT='D:\victory-data\profile-X.txt'`:
  sampling profiler, report and `.folded` stacks rewritten every 30 s,
  cumulative from the start.
- `VICTORY_GPU_TIMING=1 VICTORY_D3D12_DRED=1`: each list run alone and
  timed on the card ("the card's time by command"); slow, changes pacing.

Other machines' work (cargo builds of agents) skews absolute numbers.
Compare runs only on a quiet machine, or compare ratios.

Runs go different ways (the title screen comes at a different second, the
scripted presses land elsewhere, the user walks the character): two runs
are seldom the same scene. To measure a switch, give it an `alt` mode
(twenty seconds on, twenty off, each change logged; see
`VICTORY_RELAY_WIDE`, `VICTORY_WIDE_FILLS`) and split one run by phase:
`python scripts/altphase.py <log> "<change line text>" "<word of the ON
line>"` (usual frames per 1,000 draws, the fence's way, the relay's
copy-back). Add the switch's module to `VICTORY_LOG` so the change lines
are kept. A run that ends long before its timeout with nothing in the log
was ended from outside (window closed, title bar clicked): add
`victory_hle=info` to see "window closed".

## 4. Read the numbers

- Frame line (`victory_gpu::host`): `N fps, X ms/frame (preparing: copies
  .. textures ..; on the GPU; copying back ..), draws/frame`.
- Frame log (`victory_gpu::framelog`): every 5 s, `usual frame` and each
  `long frame` with the submit thread's work, card idle, recording calls,
  texture refreshes, remappings, page/guard faults.
  `python scripts/framelog.py <log> [last N]` tabulates them.
- Indirect clamps: `indirect commands per frame by their clamp` (draw.rs).
- Textures read past their guards: `textures read past their guards`.
- Profile, gameplay only: copy the `.folded` once the world has run a
  while, then diff with the final one:
  `python scripts/folddiff.py t1.folded t2.folded <frames between> <role>`
  (roles: submit, record, busy, late-wait, title-main, pipelines; frames
  from the reports' "frames presented meanwhile" lines). Stacks under
  `blake3_*` are cut (no unwind data): find callers in the source.
- Captures: `python scripts/sheet.py <cap dir> <out.png> <n> [first frame]`
  for a contact sheet; `compare.py` lines two runs up on their world frame.
  Always look at the pictures after a change that touches what is drawn.

## 5. Report

Say: build (commit), switches, scene (frames after the world, draws a
frame), median and worst frame, what changed against the reference run,
and whether the pictures are right. Add the result to MOONSHOT.md's log.

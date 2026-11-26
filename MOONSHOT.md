# Moonshot: One Memory

The console has one memory. The CPU and the GPU read and write the same
bytes at the same addresses. This fork keeps two: title memory, and host
copies the card can see. It then spends most of every world frame keeping
the two in step. The moonshot is to stop keeping two. The card works on the
title's own bytes, and the copy model goes away for everything except
textures.

Four days with unlimited people. Six phases. Each phase is something you can
measure, and each one leaves the tree working.

> **Outcome (2026-11-25, after all six phases were run).** The machinery
> works and is tested. The title runs on a committed page pool, view memory
> can be placed, the raw-address directory has mirrors, and flights know
> what in-place shaders write. **But the bet failed on this driver.** Memory
> opened in place is not cached by the card, so hot buffers cost up to 145
> times more to read, and world frames got 20-40 % slower. Also, nothing in
> the title's view memory can be placed, because its pages are scattered or
> aliased. In place therefore stays off by default.
>
> The same measurements found the lever that does work: **the card's own
> memory**. It reads scattered data 28 times faster than today's host
> copies. Giving copies of title buffers (from 1 MiB) and windows room there,
> sized to the card, is now the default on cards with 16 GiB or more. On the
> RTX 3090 that took play from 521 to 342 ms per frame (1.9 to 2.9 fps), and
> the opening cutscene from 974 to 703 ms. See "Results" at the end.

## The bet, in numbers

A world frame at today's defaults (run log `victory-step4.log`, 2026-11-24):

| Per frame | Today |
|-----------|-------|
| submissions | 31.7 |
| write-watch askings | 956, **537 ms**, 4.9 GB checked, of which 4.7 GB had not changed |
| refreshes of host copies | 9,625 (10.4 GB of ranges considered) |
| title copies | 7,515 |
| page faults | 450,163 |
| moving copies around | 42.7 ms, 98 MB unmapped |
| frame time | 240-450 ms (2-4 fps) |
| resident memory | about 24 GB |

Almost all of this cost is on the CPU, and almost all of it exists to answer
one question: what did the title write since the card last looked? When the
card looks at the title's memory directly, nobody has to ask that question.

The card does not lose anything by this. Today's host copies already sit in
the PC's memory (a custom heap, write-back, pool L0), so the card already
reads buffers over PCIe. The probe below shows it reads a heap opened on
title memory exactly as fast.

## What phase 1 already proved (2026-11-25, on this PC's card)

`cargo test -p victory-gpu --lib -- --ignored card_probe --nocapture`, on the
NVIDIA GeForce RTX 3090 in this machine:

- Private title memory can be opened in place, either the whole allocation
  or a part of it. The card reads the title's bytes, including bytes the CPU
  wrote afterwards with no call in between. The CPU reads what the card
  wrote, whether a copy or a shader wrote it.
- A committed section (the physical page pool, if it were made committed)
  can be opened in place. A buffer placed at a view's offset shows the
  view's bytes. The card's writes land, even through a read-only view, and
  even after the view is unmapped.
- A reserved section, which is how `xmem` makes the page pool today, is
  refused (`0x80070057`). **This is the one thing that blocks view memory
  from going in place.**
- Cost of the card copying out 512 MB: 20.9 ms from a section heap, 20.8 ms
  from a private heap, 21.1 ms from today's kind of host copy. **The GPU
  bandwidth is identical.**
- Cost of opening: 27.5 ms per 512 MB for a section, 99.7 ms for private
  memory. Opening makes the pages resident (working set +512 MB), so the
  pool must be opened in chunks, as they are used, not all at once.

## The target at the end of day 4

- World at 15 fps or better on the test machine (stretch goal: the title's
  own 30). Today it runs at 2-4.
- Write-watch askings and refreshes for buffers fall by 90 % or more. Buffer
  copy-back falls to zero.
- Resident memory falls by the size of the copies, aiming for 14 GB or less.
- Every page the title has mapped can be read by raw address from shaders.
  This is what the ray tracer needs: its traversal follows 40-bit pointers
  through title memory, and a page with no host copy is lost today.
- The copy model stays as the fallback (`VICTORY_INPLACE=0`), and for
  textures, whose tiled layouts need a host copy anyway.

## The plan

### Phase 1: Ground truth (day 1, morning)

Run the probe on the card, then take a census of the world's buffer memory
by kind (private committed whole, one run of a section, scattered pages) and
take a baseline frame account.

- Done: `card_probe_*` (an ignored test) and the numbers above.
- Exit: these numbers in this file. Census in phase 3's run.

### Phase 2: A page pool the card can open (day 1)

`XMemAllocatePhysicalPages` makes one reserved section per call today (the
title asks for 6.1 GiB in one call). Instead, it splits the pool into
committed chunk sections of 256 MiB. Each chunk is made the first time one
of its pages is mapped, so commit grows with use, as it does today, and the
6.1 GiB never has to be committed at once. A view that crosses a chunk line
becomes two views. The GPU side opens a chunk as a heap the first time a
buffer in it is placed, and keeps that heap: one opening per chunk, not one
per buffer.

- Exit: unit tests for page numbering and splitting views at chunk lines; a
  card test showing that a view's bytes are read through its chunk's heap.
  The workspace tests pass.

### Phase 3: Every buffer in place (day 2)

`inplace::place_for` takes view memory: when the range is one run of one
chunk (`Cover::Run`), a buffer is placed in that chunk's heap at the run's
offset. Everything downstream already works on placed buffers
(`Shadow::placed`): descriptors, root views, vertex and index buffers,
indirect arguments. No refresh, no watch and no copy-back is done for them.
Scattered and holed ranges keep copies.

- Exit: in a world run with `VICTORY_INPLACE=2`, the in-place line shows
  view memory placed. Watches and refreshes per frame fall, measured against
  the table above.

### Phase 4: Shaders read any title page (days 2-3)

The raw-address directory (one entry per 64 KiB of title address space)
points into the same heaps: private parts and pool chunks. This works for
every mapped page, not only those that happen to have a host copy. With
this, the ray-tracing builder no longer needs its quarantine copies.

- Exit: a card test where a shader loads, through the directory, from a pool
  view and from private memory, and sees the title's bytes.

### Phase 5: The card keeps its own time (day 3)

Bottom-of-pipe memory writes (the title's fences) that land in in-place
memory are written by the card itself (`WriteBufferImmediate`), instead of
by our relay thread after a wait. With buffers in place, nothing has to be
copied back after a submission, so late wait becomes safe, and the
submitting thread stops waiting for the card. On the console, the title
already synchronises with the GPU through exactly these memory values.

- Exit: the submitting thread's waits per frame fall to near zero, with no
  new black or broken frames in world captures.

### Phase 6: Flip, cut, light (day 4)

- Defaults: in place on. `VICTORY_INPLACE=0` brings back the copy model.
- Watch work for in-place ranges is removed.
- With title memory whole, the BVH builder is tried for real
  (`VICTORY_BVH_BUILDER=1`).
- The README status table is updated with measured numbers, not hopes.

## Risks, and what we do about each

| Risk | Answer |
|------|--------|
| Opening a heap pins its pages | Open chunk by chunk, on first use; cap the total with `VICTORY_INPLACE_MB`. |
| While a heap stands, the title cannot free or protect pages in it (error 87) | The title's heaps are allocated once and kept. `title_memory_freeing` lets a heap go first; pool views can be unmapped under a section heap (the probe shows it). |
| Tile mappings into an opened heap remove the device | Tiled resources keep copies. The tile path never takes an opened heap. |
| Committing the pool fails on a busy host (error 1450 today) | Chunks are committed as they are mapped. If a chunk cannot be made, `xmem` falls back to a reserved chunk, and that chunk keeps copies. |
| Hot buffers in the PC's memory are read too often | Card copies (`VICTORY_CARD_BUFFERS`) stay for the big ones, as today. The placement rule already leaves them out. |
| The title reads memory a submission not yet run will write | Same as on the console: the title waits on its own fences. Phase 5 makes those fences the card's own writes. |
| The same pages shown at two addresses | In place, this is no longer a problem: both addresses are the same bytes to the card too. |

## What it unlocks beyond the frame rate

- **Light.** Ray-traced bounce light and shadows need traversal through
  arbitrary title memory. Phase 4 gives the traversal that memory.
- **Memory.** Gigabytes of host copies, and their "before" snapshot twins,
  disappear.
- **Less code.** Much of `shadow.rs` (3,300 lines) and `watch.rs` (1,200
  lines) exists to keep two memories in step. Once in place is the default,
  that becomes the fallback path, not the hot path.

## Results

World runs on this PC (Windows 11, RTX 3090 24 GB), the diagnostics build,
the same save, and keys scripted (`VICTORY_KEYS=30:Enter,35:X`). Each run
is lined up on the first frame with more than 500 draws. "Cutscene" is the
opening cutscene (frames with more than 1,000 draws, 5-150 s after that
frame). "Play" is the first gameplay after it. One run each, so a few
percent either way is noise.

| Run | Switches | Cutscene frame / CPU / GPU (ms) | Play frame / CPU / GPU (ms) |
|-----|----------|--------------------------------|-----------------------------|
| 2 | `VICTORY_INPLACE=2` | 1,161 / 421 / 533 (first 90 s) | not reached |
| 3 | none (old defaults) | 974 / 515 / 269 | 521 / 257 / 147 |
| 4 | `VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_CARD_BUFFERS_MB=4096` | 818 / 428 / 229 | 444 / 208 / 136 |
| 5 | run 4's and `VICTORY_CARD_WINDOWS_MB=4096` | **703 / 334 / 215** | **342 / 162 / 120** |
| 6 | none (new defaults: `shadow::card_room`, budget 23,557 MB, so 3,926 MB each) | 734 / 387 / 207 | not reached in the run's 5.5 minutes |

What changed by default: `shadow::card_room`. On a card whose budget is
16 GiB or more, copies of title buffers from 1 MiB on, and windows outside
buffers, may take up to a sixth of the budget each in the card's memory.
`card_for` and `window_card_for` still keep a fifth of the budget free for
textures. Smaller cards keep the old numbers (16 MiB, 512 MB, no windows),
because the change was not measured on one. Every switch still overrides.

What stays in the tree, off by default (`VICTORY_INPLACE`): the committed
pool chunks, view placement, directory mirrors, flight declarations, and the
probes that explain why (`card_probe_*`, `card_view_memory_goes_in_place`,
`card_heap_as_the_pool_probe`, `card_gathers_from_memory_in_place`; run them
with `cargo test -p victory-gpu --lib -- --ignored card_ --nocapture`).
They become useful on a driver that caches memory opened from an address
or a file mapping, which can be retested in one command.

What is next, by these numbers. The CPU's preparing is still half of the
frame: textures alone take 110-400 ms of it. Ray-traced light is untouched
by this work; its traversal pages are private memory, which today's copies
already reach.

## Progress log

- 2026-11-25. Phase 1: card probe added and run (results above).
- 2026-11-25. Phase 2, code and tests. `xmem` makes the pool of committed
  chunk sections of 256 MiB whenever the GPU side puts memory in place
  (`VICTORY_POOL_COMMIT=0/1` overrides). Chunks are made at first map, and a
  chunk that cannot be committed falls back to reserved. The GPU side opens
  each chunk once (`inplace::section_heap`). Tests:
  `a_committed_pool_maps_across_chunk_lines` (xmem), and
  `warp_view_memory_goes_in_place` / `card_view_memory_goes_in_place`. All
  pass, on the RTX 3090 too.
- 2026-11-25. Phase 3, code. `inplace::place_for` places view memory that is
  one run of one committed section, and that no other address shows, in the
  section's heap. When a second address starts showing those pages, what is
  in place there is let go. The in-place line in the log now counts view
  placements and pinned memory.
- 2026-11-25. Phase 4, code. The raw-address directory gives view memory of
  a committed section a mirror: one buffer over the whole section, and every
  page of the section's views given it as views are mapped, not as shaders
  fault. When a view goes, its entries are cleared, so they are no longer
  left stale. Pages a host copy holds stay with the copy. Fixed: offsets
  into buffers in place were short by the bias when the title memory did
  not begin on a 64 KiB line.
- 2026-11-25. Phase 5, code. With buffers in place and submissions sent
  ahead, the ranges written by in-place shaders are declared to the flights
  (nothing is copied back), so our own reads of that memory still wait. This
  removes the "do not combine `VICTORY_INPLACE=2` with late wait" rule. The
  card writing the title's fences itself (`WriteBufferImmediate`) is left
  for later: it would change when the title sees its fences, and the
  flights' ordering rules depend on that.
- 2026-11-25. World run 2 (`VICTORY_INPLACE=2`, the pool committed): the
  title boots and plays on 29 committed chunk sections. Private memory in
  place: about 3,350 title buffers and 80 windows (568 MB). **View memory:
  nothing placed.** Of 626 asks, 457 are scattered or holed (the title maps
  its physical pages out of order) and 169 are shown at two addresses
  (tiled heaps). No section had to be opened. A placed buffer needs
  contiguous pages, and the title's view memory has none.
- 2026-11-25. Phase 3b probe (`card_heap_as_the_pool_probe`): could the
  pool be heaps the card makes, which can take tile mappings (the
  console's page tables)? No, on this driver. Every shared heap the CPU
  could see is refused (`E_INVALIDARG`), and the handle of a shared heap
  the CPU cannot see is not a section (`ERROR_INVALID_HANDLE`). Scattered
  view memory therefore keeps copies for buffer views. Raw addresses still
  reach it in place, page by page, through the directory's mirrors.
- 2026-11-25. World runs 2 and 3 (same binary, in place against defaults,
  lined up on the moment the world starts, the same cutscene): with
  `VICTORY_INPLACE=2`, CPU preparing is about the same (300-640 ms a
  frame), copying back is a little lower, and **the card's time doubles**
  (about 450-630 ms against 220-300). Frames are 20-40 % slower.
- 2026-11-25. Why: `card_gathers_from_memory_in_place`, on the RTX 3090.
  Shader reads timed per kind of memory:

  | Memory | scattered over 256 MB | over 1 MB (cache-sized) | copy 256 MB |
  |--------|----------------------:|------------------------:|------------:|
  | today's host copy (committed, the PC's memory) | 61.9 ms | **0.4 ms** | 11.9 ms |
  | the card's own memory | **2.3 ms** | 0.3 ms | 0.8 ms |
  | private memory in place | 62.6 ms | **58.1 ms** | 10.6 ms |
  | a committed section in place | 62.6 ms | **58.0 ms** | 10.6 ms |

  **Memory opened in place is not cached by the card.** A buffer placed in
  an opened heap must be cross-adapter, and the driver bypasses the cache
  for it. Data the card reads again and again, which is what most buffers
  are, costs about 145 times more. Phase 1's probe measured only copies,
  which stream and do not use the cache, so it did not show this. **The
  bet "in place costs the GPU nothing" is false on this driver.** In place
  stays off by default. The work stays in the tree, behind its switch,
  because it is correct and tested: committed pool chunks, view placement,
  directory mirrors, flight declarations.
- 2026-11-25. Runs 4 to 6: the card's memory for copies, then the default (Results).
- 2026-11-25. Run 6's log showed what texture refreshes were doing.
  Across all threads they took 600-1,190 ms a frame, and 490-1,040 ms of
  that was waiting for title memory to hold still. One lock covered the
  whole address space, and the runtime's 1,459 remappings a frame held it
  for 778 ms of it. The lock now covers ranges (`watch::title_memory_reading`
  and `title_memory_remapping` take the memory they read or change; `xmem`
  passes the extent of the views a change touches). A change waits only
  for reads of its memory, and a read only for changes of its memory. A
  change that waits holds back new reads of its own memory, so it cannot be
  put off for ever. Run 7, defaults plus this: texture lock waiting is 0.0
  ms, texture refreshes take 51-84 ms a frame on all threads, and the
  cutscene runs at 630 ms a frame, CPU 230 (textures 112), GPU 200. Before
  this it was 734 / 387 (254) / 207. Play: 401 / 140 / 125, against
  521 / 257 / 147 on the old defaults.
- 2026-11-25. Run 7, play: the card is 28-38 % busy. The submitting thread
  waits for the card 110-118 ms a frame, and the card waits behind it
  140-200 ms. The submissions are serialised.
- 2026-11-25. Run 8: run 7's defaults and `VICTORY_RELAY=2`, the fork's
  existing switch that sends submissions ahead. Cutscene 468 ms a frame,
  CPU 252. Play **256 ms (about 3.9 fps)**, CPU 136. The card is 45-52 %
  busy, and the submitting thread waits 68 ms a frame. Its "on the GPU"
  numbers (33-59 ms) are counted differently with late wait and cannot be
  compared. The black-frame check is left out with late presents, so its
  0 proves nothing. No crash. The relay stays off by default until frames
  captured with and without it have been compared.
- 2026-11-25. Runs are now made from a copy of the saves
  (`saves-backup-20261125-after-capB`, outside the tree), restored before
  each run. The title autosaves, so without the copy each run started where
  the one before had saved. Run C (no relay) ended about 2.5 minutes into
  the prologue on a breakpoint in the title's streaming. It came right
  after a DirectStorage read straight into title memory had failed with
  ERROR_INVALID_USER_BUFFER (0x800706f8). Such reads never took the read
  lock: while `xmem` maps one page of a view anew, the view's other pages
  are a reservation for a moment, and a kernel write into them fails. Each
  request now holds, in one taking, the title memory it reads or writes in
  place (`watch::title_memory_reading_all`). One taking, not two, because
  a change waiting on the first would hold back the second. Run C2, same
  save, with the fix: more than 6 minutes in the prologue, no failed read,
  no crash.
- 2026-11-25. Runs C2 (no relay) and D (relay), same save, frames captured
  every 8th frame in the world:

  | Run | Prologue opening frame (CPU) | Play frame (CPU) |
  |-----|-----------------------------|------------------|
  | C2, no relay | 540 ms (182) | 412 ms (147) |
  | D, relay | **378 ms (190)** | **270 ms (145)** |

  The same shots show the same picture (frames +2 to +200 after the world
  starts). Neither run has a black frame, a crash or a failed read. Later,
  C2's character walked to the canteen on its own while D's stayed at the
  window. C2 then shows big flat triangles (green, black, brown) across the
  screen in nearly every frame. Those are older than this work (the owner
  saw them before) and appear without the relay. They are a bug of their
  own, still open. **The relay is now on by default** (`VICTORY_RELAY` at
  2; `0` turns it off). Unit tests keep the synchronous model they were
  written for; the two tests specific to that model skip under the relay.
- 2026-11-25. Run E, the new defaults and no switch, same save: the relay
  is on by itself. The prologue opening runs at 436 ms a frame (CPU 181)
  and play at **264 ms (about 3.8 fps)**, CPU 137. No black frame, crash
  or failed read. Play by default over the day: 521 ms with the old
  defaults (run 3), 401 with card memory and the range lock (run 7), 264
  now.
- 2026-11-25. Resizable BAR (`card_gpu_upload_heap_probe`): Direct3D's
  form of it, `D3D12_HEAP_TYPE_GPU_UPLOAD`, is not offered on this PC.
  `nvidia-smi` shows a BAR1 window of 256 MiB, so Resizable BAR is off in
  the firmware; with it on, the window covers all 24 GiB. Where it would
  help: copies in the card's memory brought up to date by the CPU writing
  into them directly, with no CPU side in the PC's memory, no copy on the
  card and no hash passes to find what changed. That fits copies shaders
  only read, such as the windows. It cannot be title memory itself: the
  CPU reads such memory uncached over the bus, and the driver picks the
  address. To retest after turning it on: the same probe.
- 2026-11-25. Unattended runs can now walk the character
  (`VICTORY_KEYS=<seconds>:<key>[:<hold>]`; W A S D and I J K L are the
  sticks, Z and C the triggers, with a real pad connected too). Run G
  walked the cell block for 1,300 frames with no flat triangles. They show
  in the yard and the canteen (run C2), which the walk did not reach.
- 2026-11-25. Where the same table points instead: for scattered reads, the
  card's own memory is 28 times faster than today's host copies, and today
  every buffer below 16 MiB, and everything past 512 MB of card copies,
  stays in the PC's memory. That is the next lever (run 4).

## Day 2: upstream merged, the card set free, 20 fps as the target

- 2026-11-25. Upstream's twenty commits since our base are merged
  (`merge-upstream`). With its texture fixes and run 253's switch set the
  picture is right again: skin, walls, light (run M). Its frame-rate work
  sits behind switches, so the merged defaults still play at 250-335 ms a
  frame (run L).
- 2026-11-25. Timing run K: of the card's ~145 ms of title commands a
  frame, ~100 ms were indirect commands, 6,632 a frame, each behind a
  clamp pass of ours with a full UAV barrier on either side: the card went
  from drawing to compute and back 13,000 times a frame. The records come
  from the card itself: a culling dispatch writes a run of them (1,024 of
  32 bytes in K's frame), then hundreds of mesh draws read one each.
- 2026-11-25. **Precise indirect windows** (`VICTORY_INDIRECT_PRECISE`, on
  with `VICTORY_INDIRECT_WINDOW=1`): each host list logs what its shaders
  may write (the ranges declared as views are bound, and commands whose
  code stores through computed addresses, `Translated::raw_stores`). A
  window of clamped records ends only when a write meets its records, not
  at every dispatch or direct draw; a record is gathered for the list's
  prelude unless something recorded before it may have written it; a
  record in a range written earlier in the list opens a window over the
  rest of that range (up to 4,096 records). Indirect dispatches and
  storing shaders are windowed too. Run N: **232 clamp passes a frame
  instead of 6,632** (3,154 commands read from windows, 2,498 from
  gathered records); the submitting thread no longer waits for the card.
- 2026-11-25. Textures whose pages are shown at several addresses were
  hashed at every check: 124 MB a frame of run N's 128. Upstream's guards
  now guard every view of a page together, so only pages with more than
  `MANY_VIEWS` views still need it (`inplace::aliased`). Texture work fell
  from 42-47 to 7-10 ms a frame (run O).
- 2026-11-25. With the card no longer the limit, upstream's
  `VICTORY_ASYNC_PRESENT=1` (measured as "no gain" while the card was the
  limit) lets the title record the next frame while our thread prepares
  this one. Play: 214 ms (M) -> 163-173 ms (N) -> **142-150 ms at 6,300
  draws (O), about 7 fps**; 147 ms in the lighter cell view of M.
- 2026-11-25, night. Two engineers' branches merged (memory tracking:
  kept answers about unmapped blocks, refresh copies spread over the kept
  threads, small fills copied in place, the merge eight bytes at a time;
  command translation: lock-free reads of made descriptor slots with
  per-slot locks, per-thread memory of title-buffer and view lookups,
  cheap hashing). Run X (quiet machine, same switches): the recording
  threads' calls fell from 306-330 to 148-171 ms a frame in all; usual
  frames 75-88 ms, but every 7-8 frames a present finds the one before it
  still in the line and runs the old way, taking the backlog (300-350 ms,
  then a 3 ms frame): about 120 ms a frame on average at ~3,400-9,500
  draws (8 fps).
- 2026-11-25, night. The low points are texture streaming: the GPU's
  page-mapping copies, one page per call, cost two system calls each
  (unmap, map) at 120-1,000 us in the running title (35 us in a quiet
  process) and 16 page faults when first read; cutscene frames of 1-3.5 s
  carried 2,000-4,800 of them (0.3-2.9 s). Views for DirectStorage's
  writes are now kept per section (`VICTORY_OWN_SECTION_VIEWS`: page faults
  in those frames ~100,000 -> ~75,000). Copies instead of views
  (`VICTORY_ALIAS_COPY=1`, 5.8 us a page) drew an empty world in run U:
  the title writes pages after mapping them; `crate::alias` now takes a
  copy again when its source is written, and writes shaders' bytes back to
  the source (run Y measures it).
- 2026-11-26, night. Corrections. The periodic 300-400 ms frames of the
  captured runs are the capture: a saved frame is always presented the old
  way (it is read back at once). Without capture (runs AB, AC, AD, AF) play
  runs a 3-frame cycle of about 3, 110-125 and 200 ms, the long one a
  present running the queue's line itself. The limit is our submitting
  thread, ~94% busy (about 44 + 44 + 221 ms of work per 3 frames): not the
  relay's copy-back (~40 ms a frame on its own thread, 1,150 pieces, 71 MB
  fetched, 31 MB actually changed) and not the card (~70% busy). Run AD
  (staging pool back on): 100-108 ms a frame; run AF (card hashes a group a
  piece, the relay merging on four threads): 109-114 ms, no gain while the
  submitting thread is the limit.
- 2026-11-26, night. Two more branches merged: the descriptor heap pass
  beside the preparing and the merge under flights on the wide pool (both
  on), shared askings for a wide refresh (off), and the GPU's page-mapping
  copies applied by a thread of ours in queue order
  (`VICTORY_ASYNC_PAGE_MAPPINGS`, off). Run AG: play at 100-106 ms a frame,
  preparing 68-84 ms. Run AI (`VICTORY_WIDE_REFRESH=alt`): no difference
  between the phases in play (100-115 ms either way). Run AH (async page
  mappings): the loading screens deadlocked (a title access to a pending
  target waited 10 s for a batch that waited for it), then the title
  crashed; sent back for a fix.
- 2026-11-26, night. Run AK (async page mappings after the fix, capture
  on): no wait warning, loading and play right, pictures as run AE's (the
  smudged face in the bar cutscene was in AE too: streaming, not this).
  World frames of a second or more: 18 (AG 30, AJ 31); the worst 1.5 s
  (AG 4.1 s, AJ 3.1 s). Play unchanged (75-79 ms usual frames, ~117 ms
  with the capture's old-way presents). The relay's landing is still
  ~60 ms a frame (AJ: 65.8 ms, 58 MB in ~11,700 kernel writes of 4 KiB):
  it is now copied in place (`VICTORY_WRITE_PLAIN`, run AL).
- 2026-11-26, night. Runs AL and AM (no capture; async page mappings,
  copy-back writes in place, the profiler following the relay's threads):
  play as run AJ, 100-110 ms a frame; world frames of a second or more 15
  (AJ 31). Writing in place changed nothing measurable: only ~590 writes a
  frame (26 MB) go through `write_mapped`, the other ~12,000 stretches go
  through the own view (`VICTORY_BACK_OWN_VIEW`). The relay is busy ~53 ms
  a frame and waits for the card the rest: 35 % in the merge loop itself,
  15 % reading title bytes, 13 % writing them. The card is ~85 % busy
  (title lists 57-62 ms, our send-and-hash 15 ms, compare 5.5 ms a frame),
  so the card is near the limit again: run AN times the title's commands.
- 2026-11-26, night. Run AN (timing run, each list alone): the title's
  commands take 57.5 ms of card time a frame in play, spread over 1,302
  kinds: dispatches 30.5 ms in 3,844, indirect commands 19.1 ms in 5,109,
  mesh 3.4 ms; the heaviest single dispatch 4.7 ms. No hot spot: whatever
  makes the translated shaders slow is systematic (a translator review is
  under way). Waits for every earlier shader write: ~710 a frame for the
  title's barrier groups, ~420 around the ~210 bounded indirect commands.
- 2026-11-26, night. Run AO: `VICTORY_PRESENTS_AHEAD=3` is worse (114-125
  ms against ~105; the title's own threads take twice the CPU). The title
  waits on no fence event of ours (`pace`: 0 waits a frame); a fence is set
  ~123 ms (1.2 frames) after the title's Signal for it. A census of the
  barrier groups (`VICTORY_BARRIER_CENSUS=1`): in menus most are buffer
  copies (made on the CPU at submission) and render target <-> shader
  resource layouts (moved by our own state tracking); in play most order
  unordered access. `VICTORY_BARRIER_PRECISE=1` records the wait only for
  groups that need it (run AP).
- 2026-11-26, night. Run AP (`VICTORY_BARRIER_PRECISE=1`, captured):
  pictures right; ~83 of ~710 group waits a frame left out in play (124 of
  967 in the cutscene), the title's lists 55-60 ms of card time against
  57-62: within the noise. Stays off. `VICTORY_ASYNC_PAGE_MAPPINGS` is now
  on wherever `VICTORY_ASYNC_SUBMIT=1` is (runs AK, AL).
- 2026-11-26, night. Translator review (static, the 50 programs of the 40
  heaviest kinds): no local memory, no waterfall loops, 2.3 DXIL
  instructions per console instruction. Systematic costs: (1) one-wave
  thread groups: 58 % of that compute time runs 32-thread groups, and an
  SM of this card holds 16 groups, so 16 of 48 warps; (2) the legacy
  multiply's zero test (13 % of vector instructions, +17 % host
  instructions, and it splits multiply-add chains); (3) wave64 programs run
  as two waves with two barriers per exchange; (4) a signed-zero fix-up on
  every min/max, half of them against a non-zero constant.

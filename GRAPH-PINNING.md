# Shared pinned pages for graph capture

## Problem

A sealed graph guards every page its recording touched (`PageGuard`): the page is never
released or relocated, and nothing new is carved from it. So each capture's warmup has to
take pages no earlier graph holds, and the memory a stream's recordings hold is the sum of
every graph's working set. A Devstral load that records ~80 widths over 1 GiB pages pins
~80 GiB and runs out of memory.

## Change

Graphs recorded on one stream replay one at a time, in stream order. The room a graph's
temporaries used is only written while that graph replays, and nothing needs it between
replays. So a guarded page's free room may be carved by another capture on the same stream.

- **Who may carve a pinned page:** a capture on the stream whose memory holds it, from its
  warmup run (`graph_prepare`) to `stop_capture`. Work outside a capture never may, because
  it outlives the next replay, which would write over it. `StreamCapture::pinned_pages()`
  answers `Share` or `Avoid`, and `Command::reserve` passes it to
  `MemoryManagement::reserve_sharing`.
- **What a capture can take:** only free slices. A handle that holds a slice (a graph's
  input, an output the caller keeps, a KV pool, a step table) keeps it out of the free list,
  so no other graph is ever handed it. A graph also keeps strong bindings to every buffer
  it writes (`Graph::written`). An output stays the graph's for the graph's life even if the
  caller drops its handle. Temporaries are freed inside the capture, so their slices go back
  to the free list, and a later capture on the stream can reuse them. No step classifies
  buffers when a capture ends. Refcounts do that job.
- **What stays as it is:** guards still keep a page from release and relocation for as long
  as any graph holds it, so a page stays pinned while any graph that touched it lives. No
  new public API. Exclusive pools and the small (metadata) pool still never share. The
  small pool holds the info buffers a capture keeps for its graph's life.

## Record the largest graph first

Only the arena's current pool carves (`PoolArena::try_reserve`). When an allocation outgrows
the pages, a pool of larger pages takes over, and the older pinned pages serve no later
capture. Recording the widest graph first grows the pages, and pins them, once. Every
smaller graph after it then fits in that room. The stream holds about the largest graph's
temporaries plus every graph's held I/O. A larger graph recorded later still works: its
warmup adds pages (or grows the page size), and only the extra is new memory.

## Outputs never sit on shared room: the rehearsal

A graph's replay writes its temporaries over the room the stream's other graphs share. A
buffer the captured work leaves behind, an output the caller reads, must never sit there, or
another graph's replay overwrites it. Whether an allocation is a temporary or an output is
only known once the work is over, and by then a recording has baked its address. So the work
runs before it is recorded:

1. **Rehearse.** `Client::capture(work)` prepares the stream and runs `work` while the stream
   rehearses. The capture state decides this, so no guard has to live between the steps: a
   launch on a stream whose capture is prepared and not yet open is
   `LaunchMode::Rehearse`. It compiles, does the launch's host side (resolving its buffers,
   building its metadata buffer and the staging that uploads it) and stops before the
   dispatch. Other streams keep running. An override that runs the stream, which is what a
   tune's measurement opens, still wins, so tunes measure for real. Pages stay eager, so the
   pools grow to what the recording will ask for. Every allocation goes through `Command::allocate`, which lists it with its
   size. When the run returns, with its output still held, `graph_rehearsed` drains the
   deferred frees and marks each listed allocation as outlived (held) or not.
2. **Until stable.** The rehearsal repeats until two runs in a row list the same allocations,
   at most four times. A first run can take another path than the steady state (a tune, a fused
   optimization being built).
3. **Set aside.** As the window opens, one slice per output of the last run is reserved off
   the shared room (`PinnedPages::Avoid`) and held for it.
4. **Record.** The recorded run allocates the same sequence. Allocation *i* takes its slot when
   the plan has an output of that size at *i*, and carves the shared room otherwise.
5. **Check.** At seal, a recorded run that allocated another size or another count, or kept a
   buffer its rehearsal let go, does not seal. That buffer sits on shared room. An output the
   recording let go is no divergence: its slot only went back to the pools.

What this leaves to callers:

1. **The work allocates alike on every run.** It allocates the same buffers in the same order,
   and keeps the same ones. A recording that does not is refused, never silently wrong.
2. **The work does not read the device.** A rehearsal's launches never ran.
3. **Late-freed temporaries share less.** Burn fusion releases handles lazily. A temporary
   still held when a run ends looks like an output, takes room of its own and is not shared.
   This is safe and only costs memory, so drain fusion before each run ends.
4. **One stream, one order.** Graphs on different streams never share room, because each stream
   reserves from its own memory.

The graph still holds strong bindings to every buffer it writes (`Graph::written`), so an
output stays its graph's even if the caller drops it.

## Metabolic follow-up (separate change)

- Record widest first. `PagedBatchStepper::record_every_width` records decode, then
  prefill. `RecordingList` walks `DecodeWidths::rows` in ascending order
  (`sizes_up_to_most`), and `PagedPrefiller` walks widths in the order groups first ran at
  them. Record prefill first, since its widest step has the largest working set, and sort
  each list by descending width.
- Drop the record-within-room stopgap (#1530), which recorded narrowest-first and stopped
  once the room ran out.
- Capture through `Client::capture` (or burn's `capture`, which calls it). The rehearsal
  replaces the warmup. `CapturedPrefill`'s copy of its logits after each replay is no longer
  needed for correctness.

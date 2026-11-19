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

## Invariants for callers

1. **Keep a graph's outputs held for its life.** Cubecl already ensures this for buffers
   the graph writes, through `Graph::written`.
2. **An output allocated inside a capture window is valid only until another graph on the
   stream replays.** This covers the recorded run's own outputs and anything the warmup
   allocates and keeps. A capture may carve such a buffer from the room an *earlier* graph's
   temporaries use, and that graph's next replay writes over it. Two ways to stay safe:
   - Allocate the buffer before `graph_prepare`. An eager allocation never lands on pinned
     room.
   - Read or copy it right after its own graph's replay, before replaying another graph on
     the stream.

   The reverse direction is safe. An earlier graph's held output is live during every later
   capture, so no later capture can take it.
3. **Late-freed temporaries share less.** Burn fusion releases handles lazily. A temporary
   still held when a later capture runs is not free, so that capture cannot take it. This is
   safe and only costs memory, so flush fusion before `stop_capture`.
4. **One stream, one order.** Graphs on different streams never share room, because each
   stream reserves from its own memory.

## Metabolic follow-up (separate change)

- Record widest first. `PagedBatchStepper::record_every_width` records decode, then
  prefill. `RecordingList` walks `DecodeWidths::rows` in ascending order
  (`sizes_up_to_most`), and `PagedPrefiller` walks widths in the order groups first ran at
  them. Record prefill first, since its widest step has the largest working set, and sort
  each list by descending width.
- Drop the record-within-room stopgap (#1530), which recorded narrowest-first and stopped
  once the room ran out.
- I/O is already outside the window. `CapturedDecode` allocates `tokens`, `ngram_rows`,
  sampler rows, the counter and the seed before recording, and its output is the
  token buffer it writes in place. `CapturedPrefill`'s output (logits) *is* allocated inside
  its window. Each `step` copies it (`narrow(..).add_scalar(0.0)`) on the capture stream
  straight after the replay, so invariant 2 holds. A refactor that reads the logits later
  must keep that copy.

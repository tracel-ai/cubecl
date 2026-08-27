# Error handling in the CubeCL runtimes

This document describes where stream error handling stands, what is structurally wrong with it, and a design that fixes those problems while adding the capability the current model is missing.

## Where it stands

Errors are lazy.
A failed launch is queued on the pooled backend stream and surfaces at the caller's next flush, sync, read, or profile end.

Several logical streams share one pooled stream, because `stream_index` folds the id space onto `max_streams` slots.
So a queue entry carries the logical stream that caused it, and only that stream's flush takes it.
Errors a slot cannot attribute go in unattributed and surface on whichever stream flushes next.

Attribution answers who reports a failure.
It does not answer the question a read has to ask, which is whether the bytes it is about to copy were ever written.
A handle names where a buffer was created and nothing re-tags it, so no stream id can answer that.

The current answer is that failures name the buffers they left unwritten, and a read asks every pooled slot whether any queued error names the buffers it is about to copy.
The claim on a buffer outlives the report of the error that made it, and ends when something writes the buffer again.

## What is wrong with it

**The invariants are held by convention, in five places.**
Every backend has to remember to call `ensure_written` on a read, to name `memory_ids_written` when a launch fails, to call `written` when one succeeds, and to check `is_skipped` before naming anything.
Nothing enforces any of it.
That is why the CPU and Metal backends were silently missing their `written` calls, why wgpu and Metal named buffers for dry runs that never touched them, and why wgpu had two different answers for one rejected write.
A single review pass over carefully written and already reviewed code turned up eight real defects, and one of the fixes for those introduced a fresh ordering hole of its own.
That rate is a property of the design, not of the people working on it.

**Whether a buffer can be trusted is stored in the wrong place.**
It lives as a claim inside an error entry, in a queue that belongs to a stream, and the question is about memory that belongs to no stream in particular.
Everything awkward follows from that mismatch.
A read scans every initialized slot, every entry in it, and every buffer each entry names.
A read has to be told whether its own flush will drain the queue it is asking, so that an entry is not reported twice.
Claims have to be stripped from entries at the right moment relative to a flush that the caller was never told about.

**The caps are silent doors back to the failure the system exists to prevent.**
Past `MAX_OWNED` an entry loses its attribution.
Past `MAX_REPORTED` an entry is dropped, and a buffer that nothing ever wrote starts reading back clean.
Both are documented and bounded, and both end in exactly the outcome the whole mechanism was built to avoid.

**Nothing propagates.**
This is the largest gap and it is described in its own section below.

**The machinery is invisible to callers.**
`ServerError` never says which buffers a failure invalidated.
An application that catches an error at a flush cannot map it back to its own handles, so the only way to find out that a tensor is bad is to read it and fail.

## The reframe

Whether a buffer can be trusted is a property of that memory, not a claim held by an error.

That fact belongs to the allocation, and so does its lifetime, which is what decides where all of this lives.
It has nothing to do with which stream is holding an error or whose turn it is to report one.

One vocabulary throughout, since three of them would drift apart.
A buffer **carries a failure**, or it does not.
Putting one there is to **taint**, and a buffer in that state is **tainted**.
There is no separate adjective for the state, because the state is whether the field holds an id.

### The state lives on the slice

An allocation can carry the id of the failure that tainted it, stored on the thing that already represents the allocation.

That is `Slice`, in `memory_pool`.
One type serves all four pool kinds, and it already holds the handle, the storage, the padding, the cursor, and whether the page is mapped.

```rust
pub(crate) struct Slice {
    pub storage: StorageHandle,
    pub handle: ManagedMemoryHandle,
    ...
    /// The failure that tainted this allocation, if any.
    pub failure: Option<FailureId>,
}
```

With `FailureId(NonZeroU32)` the niche makes `Option<FailureId>` four bytes, so the slice grows by a word and nothing allocates.

There is no map from memory id to failure, and nothing on the read path looks anything up.

### Nothing new on the hot path

Both paths that need this already hold the slice.

A read resolves its binding through `get_resource`, which walks pool, page, and slice to produce the storage handle.
A launch does the same for every resource it is given.
So checking whether a buffer can be trusted is a field read on a struct that had to be fetched anyway, and tagging one is a field write on the same.

Even a failure that never gets as far as resolving a resource, a compile error for instance, pays almost nothing.
`ManagedMemoryDescriptor` caches its `MemoryLocation`, which is the pool, page, and slice index, so reaching the slice from a binding is array indexing rather than hashing.

The map that remains is `FailureId` to `Failure`, on the server.
It is read when an error is built for a caller, and written when a tagged slice is released.
Neither is on the path a working program takes.

### Finding the slice from another stream

A `BufferBinding` names the stream the buffer was created on, and nothing re-tags it.

That fact is what makes stream ids useless for attribution, and it is exactly what makes them a reliable index to the manager that owns the allocation.
The resolution already exists and runs on every launch, `streams.get(&binding.stream).memory_management`.
`ResolvedStreams::get` hands back `&mut`, so a launch failing on one stream can tag a buffer owned by another without any new plumbing.

### Reuse is where the tag must be cleared

A slice outlives the allocations that use it.

Every pool reserves by finding a free slice and cloning its handle, and then `bind` replaces that handle with a freshly minted one.

```rust
// memory_page.rs:100, direct_pool.rs:287, exclusive_pool.rs:278, persistent_pool.rs:246
slice.handle = new;
```

So a `ManagedMemoryId` a caller can observe is unique to its allocation, and the doc comment on `ManagedMemoryBinding::id` saying so is accurate.
The recycled id lives only on the throwaway handle the reservation returns, which `bind` supersedes before anything sees it.

What that means here is that a tag on a slice describes the allocation that has just ended, and the slice is about to be handed to another one.
`bind` is where that happens, in four pools, and clearing there is one line.

```rust
impl Slice {
    /// Take on a new allocation. Whatever tainted the last one says nothing
    /// about the next.
    pub(crate) fn bind(&mut self, handle: ManagedMemoryHandle, graph: &mut ErrorGraph) {
        graph.untag(self.failure.take());
        self.handle = handle;
    }
}
```

An earlier draft of this document argued the state had to live on the slice because ids were reused and a map keyed by id would confuse two allocations.
That was wrong, `bind` keeps the promise, and a map keyed by id would have been sound.
The two arguments that carry the design are the ones above: the slice is already in hand on both hot paths, and it dies when the allocation does.

### Cleanup is not one place, but it is the right one

There is no handle-drop event anywhere in this codebase to hook.
A pool discovers that an allocation is dead by polling `Arc::strong_count` on its descriptor, which is a thing it does while walking its own slices.

The walking happens in more places than one, and this matters for where `&mut ErrorGraph` has to reach.

- `SlicedPool::try_reserve` calls `MemoryPage::coalesce` on every reservation attempt, which drains the slice list and rebuilds it, minting merged slices and dropping free ones. A tag on a free slice disappears there, on the allocation path, not on a sweep.
- `DirectPool::release_free` tombstones free slices and is called from `alloc`.
- `DirectPool::cleanup`, `SlicedPool::cleanup` and `PersistentPool::cleanup` all return early unless the cleanup is explicit. Only `ExclusiveMemoryPool` acts on the periodic sweep.

So a slice sheds its tag in five places, not three: written again, rebound to a new allocation, coalesced away, tombstoned, or swept.
All five are inside the pools, which is the point.
None of them is somewhere a backend has to remember anything, and none of them can be reached without the graph in hand once the graph is threaded through `reserve` and `cleanup`, which is where `coalesce` and `release_free` are called from.

A tag lost on a free slice costs nothing, since nobody holds a handle to a free slice and nobody can read it.
What it costs is the decrement, and the decrement is what the next section is about.

### The error graph

What a failure id means lives in one place, device-wide, because a launch failing on one stream can taint a slice owned by another and both have to point at the same thing.

```rust
pub struct FailureId(NonZeroU32);

/// Every failure the device is still holding, and what each one caused.
pub struct ErrorGraph { /* nodes keyed by FailureId */ }

struct Failure {
    error: ServerError,
    tagged: u32,        // how many slices carry this id
    occurrences: u64,   // how often this same failure has happened
    /// The launches this failure stopped, capped to the most recent.
    skipped: Vec<Skipped>,
    skipped_total: u64,
}

struct Skipped {
    kernel: KernelId,
    needed: ManagedMemoryId,
    produced: SmallVec<[ManagedMemoryId; 2]>,
}
```

A skip records a `Skipped` and nothing else.
It creates no node, constructs no `ServerError`, and therefore captures no backtrace, because the failure that explains it already has one and a skip has nothing to add to it.
That matters more for what it does not do than for what it does not store.
`BackTrace::capture` defers symbolizing, which is the expensive half, but it still walks the stack and allocates the frame list every time it is called.
Skips are the frequent event here, since a loop carrying a tainted buffer forward skips on every iteration, and they now cost no unwinding at all.

What that gives up is the source location of the skipped launch.
A report can name the kernel and not the line that launched it, which is the right way round: in a loop it is the same line every time, and the config-gated full history can capture sites for anyone deliberately looking.

It is a graph and not a list because that is the shape of the thing.
A failure stops launches, those launches leave buffers tainted, and those buffers stop more launches.
Assembling a report is a walk, and every walk starts from the one piece of information a resolution failure already has, the id on the slice.

A node is dropped when `tagged` reaches zero, which is a reference count without atomics, because a device thread owns the whole structure.
So the graph prunes itself, and what it prunes is exactly the failures that nothing can still read.

Keeping the id opaque is what stops this from spreading.
`Slice` gains a four-byte field and nothing else.
It does not learn what an error is, it cannot report one, and it has no opinion about streams.

The one wrinkle this creates is the decrement, and it has a trap in it.

Tainting is `set`, never `add`.
A slice that already carries a failure and is tainted again must release the one it held before taking the new one, and must do nothing at all when the id has not changed.
Get that wrong and a loop failing the same way every iteration increments `tagged` against a single slice forever, the node never reaches zero, and the invariant this whole design rests on is broken by the most ordinary program there is.

```rust
fn taint(&mut self, slice: &mut Slice, failure: FailureId) {
    if slice.failure == Some(failure) {
        return;
    }
    self.untag(slice.failure.replace(failure));
    self.node_mut(failure).tagged += 1;
}
```

The decrement itself has to be immediate, not collected into a list drained later.
A loop that taints and rewrites the same buffer without a cleanup in between would otherwise hold a node per iteration, every one of them at `tagged` zero and unreachable, retained only because nothing had got around to saying so.

So the shedding paths take `&mut ErrorGraph`.
That is a parameter through `reserve` and `cleanup`, which the server already drives, and it reaches `coalesce` and `release_free` from there.

### How large the graph can get

A node lives only while some slice carries its id, so the count is bounded by the live slices.

```
nodes <= distinct failures carried by live slices <= live slices
```

Holding a hundred thousand nodes means holding a hundred thousand tainted buffers, which is the program's own memory and its own decision.

The count is not what to worry about though.
A node carries a `ServerError` and its backtrace, so it is hundreds of bytes at least, and a program whose buffers are small can end up with error state larger than the memory it describes.

Deduplication is what makes that a non-issue, because the way a program reaches a hundred thousand failures is by failing the same way a hundred thousand times.
A loop launching a kernel that will not compile produces one failure that happened repeatedly, not many failures.
So insertion hashes the error's identity, the kernel and the variant and the message, and either creates a node or bumps the occurrence count on the one already there.
`tagged` then counts live tainted buffers and the occurrence count records how often it happened, and the pathological loop costs one node.

Deduplication needs a key that does not exist yet.
`ServerError` carries no kernel id, derives neither `Hash` nor `PartialEq`, and embeds a `BackTrace` that is not comparable.
So this costs an identity on the error type, a kernel id it can carry and a hand-written comparison over the parts that identify a failure rather than the parts that describe one.
`KernelId` already exists and is `Hash`, so the work is in the error type, not in finding something to key on.

Deduplication also collapses what is stored, not what is captured.
A backend builds its `ServerError` at the raise site, backtrace and all, before the graph ever sees it, so a repeat costs a capture that is then discarded.
That is fine for failures which are genuinely distinct and therefore rare.
It is worth fixing upstream for the case that motivates all of this, a kernel that will not compile being launched in a loop, where the compilation cache already keys by `KernelId` and could hold the failure alongside the successes, so the second launch never reaches a capture site.

If a hard ceiling is wanted on top of that, it caps the detail and never the taint.
Evicting a node leaves its slices tagged with a sentinel, so a read still fails, with a report that says the details were dropped instead of carrying a backtrace.
That is a different kind of cap from the one this design removes.
`MAX_REPORTED` dropped the taint, and bytes nothing had written started reading clean.
Nothing here lets correctness depend on retention.

### One scope around the writes

The convention problem needs a shape that makes forgetting loud instead of silent.

```rust
graph.while_writing(&args, &count, launch_mode, |server| {
    // the backend does its device work, and may return early anywhere in here
})
```

Entering taints every buffer the launch writes, and reads the inputs while it is there.
Leaving clears them if the launch was enqueued and none of the inputs carried a failure, and leaves them carrying the right one otherwise.
The default is tainted unless proven written, which is the opposite of today, so a path that forgets produces a loud spurious read failure that a test finds rather than silent garbage that nothing finds.

Entering has nothing to taint with, because the failure does not exist yet.
So entry mints a provisional node whose error says the launch was torn down before it could report, and exit replaces it with the real failure or releases it on success.
The provisional node is not a formality.
The closure can panic, staging and uniform allocation do exactly that on every backend, and a panic never reaches the exit code.
The provisional taint is what makes a mid-launch panic leave the write set carrying a failure instead of reading clean, which is the loud default doing its job in the one case the exit path cannot see.

The receiver cannot literally be the graph, since the graph lives inside the server the closure needs, so this lands as a server method that split-borrows.
The shape is what carries the argument, not the spelling.

It is a closure and not a guard value because a guard does not actually enforce anything here.
`#[must_use]` fires only when the returned value is discarded, and every path worth catching binds it and then returns early further down, which the attribute says nothing about.
There are already such paths in tree, three bare `Err(_) => return` arms in the metal launch path alone.
A `Drop` implementation cannot enforce it either, since it cannot reach `&mut ErrorGraph`, and an assertion there would fire during unwind: staging and uniform allocation panic on every backend, mid-launch, so a debug-build allocation failure would become a double panic and an abort.
A closure makes the early return structurally impossible, which is the only version of this that works.

The count is passed alongside the arguments because `CubeCount::Dynamic` holds a `BufferBinding` that is not part of `KernelArguments`, so nothing that describes the resources describes it, and it is read back from the device to decide the grid.
That is the buffer this design's own argument for skipping cites, and the obvious signature cannot see it.

A dry run enters and leaves having claimed nothing.
It was never going to write, so there is nothing to taint on the way in and nothing to decide on the way out, and treating it as an ordinary launch would either clear a buffer nothing wrote or taint one that was fine.

Host copies take the same shape, with the destination as the write set.

### What the stream queue is left with

Nothing, as it turns out.

`StreamErrors` was answering two questions with one structure, and once the slice answers the second one there is not enough left of the first to keep.
That argument is made in full further down, under what is left over, because it depends on knowing which failures taint no memory.

## Propagation

Here is what the model does not do today.

```
launch A writes out1                  fails      out1 carries the failure
launch B reads out1 and writes out2   succeeds   out2 is cleared
read out2                                        returns garbage, no error
```

Every backend clears its outputs on a successful launch without looking at its inputs.
So the model catches a direct read of a buffer nothing wrote, and misses everything computed from one.
In a fused stack that is nearly everything that matters.

Once the tag is on the allocation, propagation is a few lookups on a path that already iterates the bindings.

### Skip, do not taint

A launch whose input cannot be trusted does not run.

Running it is not merely wasted device time.
A buffer that holds garbage can be read as a `CubeCount::Dynamic` binding, or as indices in a gather, so the kernel can dispatch an absurd grid or scatter into memory that carried no failure at all.
Skipping costs the same check, because the inputs have to be read either way to decide anything.

A skipped launch leaves its outputs tagged with the failure that stopped it, exactly as a failed one does.

### Except while a graph is recording

A launch being recorded into a graph is not being executed, so there is nothing there to skip.

Skipping one would leave it out of the recording, and `end_capture` would seal a graph that is missing an operation and then replay it for the rest of the program's life.
Worse, the replay contract has the caller write fresh inputs before each replay, and that write clears the taint, so the graph goes on producing wrong answers with nothing tainted anywhere and no error on any path.

So a tainted input inside a capture window fails the capture instead.
That is not a special case bolted on for graphs, it is the rule `capture.rs` already states for a write it cannot record, that a graph missing an operation is worse than a late diagnostic, and wgpu's `end_capture` already refuses to seal when the window queued an error.
Cuda and hip do not refuse today, which is a gap that predates this design and that this rule closes.

### Replay settles like a launch

A replay writes the buffers its recorded launches were given, so it takes the same token over that write set and settles it.

Without that it can only taint and never clear.
A single failed enqueue, a transient allocation failure on the executable upload say, taints every buffer the graph writes, and none of the shedding paths can ever fire for them: the graph retains handle clones so the slices are never free, never rebound, never coalesced and never swept, and the graph itself is the only thing that writes them.
Today `MAX_REPORTED` evicts the claim eventually, which is exactly the silent door this design removes, and removing it without giving replay a settle would turn one transient failure into a permanently unreadable graph.

The graphs already keep the id list they would need, they keep the buffers they were captured against.

### The check is on what a kernel reads

A resource is checked before a launch when the kernel reads it, and tagged after a failure when the kernel writes it.
Those are two questions, not two halves of one, and an aliased argument is exactly the case where both answers are yes.

| argument | written, so tainted by a failure | read, so checked before launching |
| --- | --- | --- |
| `&[T]` | no | yes |
| `&mut [T]` that overwrites | yes | no |
| `&mut [T]` that accumulates | yes | yes |
| aliased output | yes | yes |

Checking only what is read is what keeps recovery possible.
A pure output is not read, so it is not checked, so `fill(out)` relaunches into a tainted buffer and repairs it.
Had the check covered every binding, the launch that would fix a buffer would be the one refused for its being broken.

An aliased buffer is checked, and a launch that needs one is skipped.
That is not a corner to escape from, it is the truthful answer: an in-place kernel reads what it overwrites, so it cannot produce anything trustworthy out of contents that are not.
Repair was never that path's job.
Write the buffer from the host, or launch something that writes it without reading it, and either way the contents come from somewhere trustworthy.

### The read set comes from the IR, which already computes it

An earlier draft proposed deriving it as `!writes(i)`, and that is wrong twice over.

`alias_writes` sets `writes` to true on the input an output aliases, so `!writes(i)` answers false for the very row of that table where both answers are supposed to be yes.
And `&mut` cannot tell an overwrite from a read-modify-write, which is the third row.
Accumulators, in-place reductions and fused epilogues are all `&mut` and all read what they write, and every one of them would derive as not read, so propagation would stop dead at the first `&mut` argument in the chain.

The compiler already answers this properly, and has all along.

- `PointerSource` traces a pointer back to the global it came from, through the address computations that would otherwise hide it.
- `GlobalVisibility` turns that into a `readable` and a `writable` flag per `buffer_pos`, which is the same index space the launcher registers resources in.
- `AnnotateGlobalVisibilityPass` stamps the answer onto the entry function's arguments as `BufferIOAttr::{ReadWrite, ReadOnly, WriteOnly, Dead}`, at module scope so functions that were not inlined still contribute.

It runs in the wgsl, spirv and cpp pipelines today, so wgpu, vulkan, cuda, hip and metal all have it.
It is consumed while generating code, for `const` qualifiers and `NonWritable` decorations.

It also survives further than an earlier draft of this document believed.
Every backend folds it into a `Vec<Visibility>` on its compiled representation, which `CompiledKernel` already carries as `repr`, and wgpu reads that back to build bind group layouts.
That `Visibility` is two-state, so it cannot say write-only or dead, and on wgpu the folded answer is forced to read-write unless `exclusive_memory_only` is set, because sliced pools can put two logical buffers behind one binding.
So the channel to the runtime exists and needs widening rather than cutting fresh, and the answer has to come from the IR attributes, not from what wgpu's shader kept.

So the work is to keep it rather than to build it: carry the visibility onto `CompiledKernel`, read it at the launch site, and register the pass in `cubecl-llvm`, which is the one backend that does not run it.

### Which deletes `WriteMask`

`WriteMask`, `arg_writes`, `alias_writes` and the macro that emits them all go, because the IR knows better than the signature and knows it per kernel rather than per launch.

The aliased row answers itself: an aliased output reuses the same IR argument as the input it aliases, so there is one global, its visibility is `ReadWrite`, and the case the Rust signature structurally cannot express needs no special handling at all.
The accumulator row is the same story.
`Dead` is a third answer the signature cannot give, and a buffer that is neither read nor written needs no check on the way in and no taint on the way out.

A kernel that fails to compile has no IR and therefore no visibility, and the right answer there is to taint every buffer the launch was given.
That is what `WriteMask` already means when nothing declared it, so the fallback needs nothing at all.

### The chain a read reports

A read should name the whole path, and the path has to outlive the buffers along it.

An allocation carries nothing but the id of the failure that tainted it.
What was skipped is recorded on the failure.

```rust
struct Failure {
    error: ServerError,
    tagged: u32,        // how many slices carry this id
    occurrences: u64,   // how often this same failure has happened
    /// What this failure stopped, newest last, capped to the most recent.
    skipped: Vec<Skipped>,
    skipped_total: u64,
}

struct Skipped {
    kernel: KernelId,
    needed: ManagedMemoryId,
    produced: SmallVec<[ManagedMemoryId; 2]>,
}
```

Reading a buffer reports the root error with its backtrace, then reconstructs the path by walking `skipped` backwards.
Find the record that produced this buffer, then the record that produced what that one needed, and so on.
Skipped `gelu`, needed #91.
Skipped `matmul`, needed #77.
Then #77, where `fill_f32` failed to compile.

Recording this on the failure rather than on the buffers is what makes it survive.
A record stored on an allocation dies when that allocation does, so a chain of links breaks in the middle as soon as an intermediate buffer is freed, which is the common case for a fused graph where only the last tensor is kept.
The failure outlives all of them, because `tagged` counts every allocation carrying the id and the record is only dropped once none do and it has been reported.

The cap keeps the newest records, not the oldest, and a running total alongside them.

Which end is kept decides whether the walk can start at all.
Reading a buffer looks for the record that produced it, and in a loop that skipped ten thousand launches that record is one of the last ones.
Keeping the oldest would leave the most recent buffers with no entry to walk from, and the report would fall back to the root error with no path to it.
Keeping the newest always starts, and a deep chain may reach a gap before the root, which costs nothing that matters because the root is on the node itself and never in the list.

Keeping a few from each end with a marker for the gap, the way a truncated stack trace does, is available if the middle ever turns out to be worth reading.
Full history belongs behind a config flag, for when someone is deliberately debugging.

## What is left over

Moving the tag onto the slice raises the obvious question: what is left that a flush would report, and does it justify keeping a queue.

The answer is that it does not.
Here is everything in the current code that queues a failure naming no buffer.

**A device fault with no owner.**
Three sites, and each one's own comment already says what it is.
Metal, on a failed command buffer, that a completed command buffer carries no logical stream.
wgpu, draining the driver's validation canary, that the driver reports these against the device and not against the launch that caused them.
Cuda and hip, in `push_sync_failure`, that the synchronize itself failed and left a context every logical stream sharing it keeps hitting.

That is one thing described three times, and it is not a per-stream queue's job.
It is a device state, and it belongs on the device.

**A caller standing right there.**
A dry run that fails to compile, which was never going to write anything, since that is the point of a dry run.
A replay handed a graph id that is unknown or already destroyed, which is a use-after-free in the caller's own code.
Both of these have someone waiting on them, and both should return, which for `replay` means changing a signature that returns `()` today.

**An open profile.**
A failure inside a profiling window invalidates the measurement, and this is the case the queue was quietly carrying.
A tunable candidate that fails to compile dispatches nothing, benchmarks at close to zero, wins the tune, and is written to a cache that outlives the process.
Nothing reads its output, so nothing would ever surface the failure, and the doc's own principle about failures with no side effect does not apply: the side effect is a fast measurement, and it is acted on.

The fix needs no queue and no new machinery, because the mechanism is already there.
`TimestampProfiler::error` exists, every backend calls it, and cuda and hip already call it straight from a failed launch whenever a profile is open, rather than through the queue.
So a failure marks every open profile at the same site where it taints memory, and `end_profile` returns that failure instead of a duration.
It records the event and not the end state, which is what makes it right: a taint that a later launch clears, and a buffer freed before the window closes, both still invalidate the measurement.

**Something that panics.**
Allocation failure is not queued anywhere today, it panics on the device thread on every backend.
Deleting the queue loses nothing there, but it is the fourth answer and the trichotomy above is not complete without it.

### So there is no queue

```
ErrorGraph       the one store of failures
Slice.failure    the index for whether these bytes can be trusted
open profiles    marked at the same site, so a measurement knows it is invalid
device fault     one Option<FailureId> on the server, for the first group above
```

`StreamErrors` goes entirely, and `Surfaced`, `Owner`, `AnyStream`, `reclaim_orphans`, `take` and `MAX_OWNED` with it.
Per-stream attribution existed to route lazy launch errors to the stream that caused them, and a launch error now taints the memory it failed to write, which needs no routing.

The health gate goes the same way, though it was never what its name suggests.
`enforce_healthy` is a flag on `resolve`, wgpu and cpu have no health gate at all, and what it actually refuses is a neighbour's flush, sync, capture or copy rather than a launch.
Refusing a neighbour for a failure it did not cause is the behaviour, and it goes because a stream is not the subject.
A buffer is, and a device can be.

`flush` and `sync` keep their signatures and report the device fault, which is the only thing left that nobody else can tell you.

### The behavior this changes

A launch that fails and whose output nobody ever reads, inside no profile, produces no `Result` anywhere.
Today a flush reports it eventually, whether or not anyone cared about the buffer.

That is the right place to land, because it is what lazy error handling already means.
A failure nobody could observe is a failure with no side effect, and under lazy reporting a side effect is the only thing that was ever going to tell anyone.

The backstop should be a log line, and that has to be built.
`ServerLogger` has methods for compilation, streaming, memory, execution and profiling, and none for a failure, so nothing anywhere logs a queued `ServerError` today.
It suits the case: a kernel that will not compile is a programming error, and a log line at the moment it happens beats a `Result` handed over some arbitrary distance later.

It also gives the error store a bound that needs no bookkeeping of its own.

> The graph leaks if and only if the program leaks memory.

A node lives while a slice carries its id, and a slice sheds the id when it is written again, rebound, coalesced, tombstoned, or released.
So a failure is retained exactly as long as some buffer is both alive and still untrustworthy, and it goes as soon as either of those stops being true.
Error retention is therefore not a bug class of its own.
A program that manages its memory has managed its error state for free, and a program that does not has a problem it can already see.

Two details this rests on.
The `skipped` history holds `ManagedMemoryId` values and not handles, so a failure's record of what it stopped never retains the memory it names.
And the device fault is the one exception, because it is not refcounted and is tied to no buffer, which is correct for a context that stays broken whether or not anything is allocated on it.

### Asking about buffers without reading them

Tainting happens when work is enqueued, so a compile or binding failure is visible immediately.
A device fault is not, because nothing knows about it until the queue is drained.

A complete answer to whether a buffer can be trusted therefore needs the barrier first, which is why it belongs on `sync` rather than on a check of its own.

```rust
client.check(&handle)?;       // instant, enqueue-time failures only, no barrier
client.sync([&a, &b])?;       // drain first, so device faults count too
client.read_one(handle)?;     // the above, and the copy
```

Three rungs that differ in what they cost and what they can see.
The middle one is a read without the read, which is what a training loop wants when it needs to know its epoch produced something trustworthy and does not want to pull it to the host to find out.

`sync` takes `impl IntoIterator<Item = &Handle>` rather than an `Option`, so the common call stays `client.sync([])` instead of `client.sync(None)`.

## What this deletes

- `entry.unwritten`, and the `Vec<ManagedMemoryId>` per entry that goes with it
- `StreamErrors::peek_unwritten` and the `reader_flushes` argument that had to be threaded into it
- `StreamPool::ensure_written` and the scan across every initialized slot
- `StreamErrors::written` and the five backends' calls to it on every launch and every host copy
- `StreamErrors` in its entirety, and with it `Surfaced`, `Owner`, `AnyStream`, `reclaim_orphans`, `MAX_OWNED`, `MAX_REPORTED` and `reclaim_reported`
- `StreamErrorSink`, and the health gate that refused work on a stream because a launch on it failed
- the ordering rule between a write and the flush that reports the failure
- the question of whether a flush and a read report the same failure twice, which no longer has two places to be asked about
- cloning `ServerError` on every read that trips over a tainted buffer
- `unwritten: Vec<ManagedMemoryId>`, the scratch field four of the five servers carry
- `flush_errors` feeding `TimestampProfiler::error`, replaced by the taint site marking open profiles directly

What replaces it is a field on `Slice`, an error graph on the server, and one token type.

## Precision

The first version taints a whole allocation, which is what the current design does.

The next step is to store the region rather than tagging the whole allocation.

```rust
struct Tainted {
    failure: FailureId,
    ranges: Vec<Range<u64>>,
}
```

That closes the hole where a host write covering part of a buffer ends the claim on all of it.
It also limits how far an over-named input can spread taint, which matters once propagation is in.
It stays a single lookup, and it costs a few bytes only on the slices that carry a failure.

A slice holds a list of these, not one, because one failure per slice cannot survive partial writes.
A buffer half-stale from failure F takes a successful launch on its other half: entry taints that half with the provisional node, and if the slot held a single failure the provisional would displace F — leaving F's untouched bytes reporting a torn-down scope that never happened.
Claims by different failures on one allocation have to coexist, each with its own ranges, and a new claim carves only the bytes it actually names out of the old ones.
The whole list lives behind one pointer, so a clean slice pays a word.

The range a claim covers comes from the binding itself: `BufferBinding` carries its `size` and both offsets, so `offset_start..size - offset_end` is computable wherever the binding is, and `StreamMemory` passes bindings down whole instead of bare memory handles.

## Diagnostics

A `FailureId` is worth showing to users, and the graph is what makes a report worth reading.

`get_resource` should close its hole while this lands.
It documents today that it asks nothing about what wrote a buffer because the check a read does costs a scan of every stream's queue.
Under this design that check is a field read on a slice `get_resource` has already fetched, so the reason is gone and the comment would be stale rather than true.

Print it in the flush report and in the read failure, so a caller can tie the lazy report to the read that tripped over it.
Right now those two are impossible to correlate.

Name the damage in the error.
`ServerError` should be able to say which handles a failure invalidated, which the error graph already counts.

Expose it on the client, where `Ok` means the bytes can be trusted.

```rust
client.check(&handle)?;
```

That is one lookup, and it lets a fusion layer or an autotuner recover per tensor instead of tearing down a device.
It is also the difference between a mechanism that protects users and one they can build on.

## Risks and open questions

**The analysis is the risk, more than the boundary.**
Today a wrong visibility costs a misapplied `const` qualifier, and this design promotes it to a correctness oracle, so its failure direction starts to matter.
It fails in the wrong direction.
An effect through a pointer `PointerSource` cannot trace is silently dropped, so the buffer keeps its default of neither readable nor writable, which stamps as `Dead`, which under this design means no check on the way in and no taint on the way out.
`MemoryEffect::ReadAll` and `WriteAll`, which inline asm produces, are explicitly ignored with a comment saying so.
And neither the analysis nor the pass has a test.
So before anything leans on it: fix the untraceable fallback, handle the asm effects, and test the pass.
The fallback turned out sharper than read-write-everything: a pointer's own type carries its address space, and for globals the binding index, so an access through an untraceable pointer is pinned to the one buffer its type names, and only a `ReadAll`/`WriteAll` from asm widens every buffer.
A non-pointer effect value — a matrix fragment, a barrier token — is register-space and touches no global, which keeps cmma kernels precise.
That hardening is independently a bug fix, since the previous default already emitted `const` and `NonWritable` on buffers it failed to trace.
One hole it does not close: the TMA ops report only their shared-memory side as effects, and the global side behind the tensor map descriptor stays invisible to the analysis — tensor maps are separate bindings outside the globals map, so the read-set step has to treat them by their own attribute, not through this analysis.

The boundary still exists and is smaller: the visibility is per kernel and cached with the compilation, while `WriteMask` is built per launch, so the launch path has to read it from the compiled kernel it is about to dispatch rather than from the arguments it was handed.
Register the pass in `cubecl-llvm` too, which is the one backend of the five that does not run it.

**Over-taint, which is smaller than it was.**
`WriteMask` over-names by argument, because a `&mut` struct of tensors registers several resources and claims all of them.
Taking the answer from the IR removes that, since visibility is recorded per buffer rather than per argument, and a resource the kernel never writes is never claimed however it arrived.

What is left is granularity.
A failure taints a whole allocation, so a launch that writes one region of a buffer taints the rest of it too, and skipping propagates that further than it deserves.
Ranges are the fix, and a full host write clears unconditionally in the meantime.

**Ranges before skip, not after.**
Clearing at allocation granularity is tolerable while it only un-fails reads.
Once skipping exists it changes control flow: a host write covering one row of a tensor clears the taint on the whole allocation, which un-skips every launch downstream, which produces garbage that carries no failure to report.
Coarse clearing plus skipping is strictly worse than coarse clearing alone, and in the silent direction.

**Getting `&mut` to a slice, and to more than one manager.**
`MemoryPool::find` returns `Result<&Slice, IoError>`, and no mutable path from a binding is handed out, so tagging needs a `find_mut` on the trait plus four pool implementations and `MemoryPage`.
`materialize` already reaches a slice mutably from a binding inside the direct and persistent pools, so the new path has a precedent to copy rather than a pattern to invent.
Reaching another stream's manager is `ResolvedStreams::get` on cuda, hip and metal, and `SchedulerMultiStream::stream` on wgpu and cpu, and both take `&mut self`, so a launch tagging buffers across several streams does them one at a time rather than holding two managers at once.
Manager counts differ too: three per stream on wgpu, two on cuda, hip and cpu, one on metal.

**The dedupe key.**
`ServerError` carries no kernel id and derives no `Hash`, so identity has to be added before deduplication can key on anything.
Check it holds for every error a backend raises: one whose message carries an address or a timestamp hashes differently every time and defeats it silently.

**Cleanup is not prompt.**
A pool only learns an allocation is dead when it walks its slices, so a tag can outlive the buffer that carried it for a while.
That is wasteful and not wrong, because the slice holding the tag is the same slice the next allocation is bound to, and binding clears it.

**Concurrency.**
The non-atomic refcount is sound because the graph and the slices are both reachable only under the device handle's mutex, not because a single thread owns them.
The distinction matters for one thing: `Arc::strong_count` on a descriptor moves asynchronously relative to a sweep, because handles drop on whatever thread the caller dropped them, and the collector thread that does exist handles only cross-stream pinned bindings on the event backends.
`Slice::is_free` already polls it and lives with that, and nothing in this design may depend on it being stable across a sweep.

**Public signature changes.**
`sync` grows a parameter on `ComputeServer` and `ComputeClient`, and `replay` starts returning a `Result`.
Both are breaks for burn and cubek, not just in-repo churn.

## Sequencing

The first step is bigger than it looks, and pretending otherwise would put the whole of the new machinery in tree alongside all of the old before anything could be deleted.

1. **`ErrorGraph`, `Slice.failure`, and the shedding paths, with `StreamErrors` cut down to reporting.**
   `unwritten`, `peek_unwritten`, `ensure_written`, `push_returned`, `Surfaced::Reported` and `MAX_REPORTED` all go in this step, because the slice answers what they were for.
   What stays is `error` and `surfaced`, so a flush still reports.
   This is a net deletion and it is testable on its own, but the read path flips on all five backends together, so it is a flag day for them either way.
   The property harness from the testing section lands here, not at the end, because the oracle is the guard a flag day needs and the dummy server it runs against waits on no backend work.

2. **The closure-scoped token, replacing the per-backend bookkeeping and the `unwritten` scratch fields.**
   After this a backend supplies device work and nothing else.
   The provisional node lands here, with a test that a mid-launch panic leaves the write set tainted.

3. **Ranges.**
   Before skipping, for the reason in the risks above.

4. **Hardening the visibility analysis, before anything trusts it.**
   The conservative fallback for untraceable pointers, the asm effects, tests on `PointerSource` and the pass, and registering it in `cubecl-llvm`.
   Independently a bug fix, and sequenced here so the next step consumes an analysis that has been made safe to lean on.

5. **The read set from the IR, then propagation, skipping, the capture precondition and the replay settle.**
   The read set comes first and brings a deletion with it, since `WriteMask` and the macro machinery that fills it are superseded by what the compiler already knows.
   The other three land together because each is a rule about when to skip.
   New conformance tests for a read downstream of a failure, for a relaunch into a tainted buffer still being allowed, for a capture refusing to seal, and for a replay recovering after one failed enqueue.

6. **Deleting attribution.**
   The device fault field, open profiles marked at the taint site, the failure log, and `StreamErrors` gone.
   Last because the profiling path has to be moved before the queue that currently feeds it can be removed.
   Burn and cubek's `sync` and `flush` call sites get audited before this step, not after, because this is where a sync stops reporting launch failures.
   The signature breaks, `sync` growing handles and `replay` returning a `Result`, ride the usual bottom-up rev bump.

7. **The chain, failure ids in messages, `client.check`, `get_resource`, and naming the damage in `ServerError`.**

8. **Give a capture its own pooled slot instead of sharing one.**
   Not strictly error handling, but it deletes a family of paths where a neighbour is refused, or briefly reported unhealthy, for a window it has nothing to do with.

## Testing

The conformance suite in `runtime_tests/stream_errors.rs` stays, and grows a case per propagation rule.

The higher-value addition is a property test over sequences of operations, run against the dummy server, with one oracle.

> A read returns bytes if and only if every byte it returns was last written by work that succeeded, and by work whose own inputs satisfied the same property.

Generate random interleavings of launch, host write, read, flush, and injected failure across several logical streams, and check the oracle after each read.
Given how many invariants this area carries, that is worth more than more example tests.
It would have caught most of the eight defects mechanically.
It lands with the first step, for the reason given there.

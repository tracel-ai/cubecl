# Error handling

A failure belongs to the memory that work left unwritten, and travels the
dataflow from there. Nothing else holds error state — no queue, no per-stream
error list, no device-wide flag.

## The model

Think of the work a program issues as a graph that is never captured or
materialised, only walked:

- **nodes are allocations** — a slice of device memory, at byte granularity
- **edges are kernels** — a launch reads some nodes and writes others

A failure flags a node. A launch that reads a flagged node does not run, and
its outputs take the same flag. That is the whole propagation rule, and two
chains that share no node are two disconnected subgraphs however they are
interleaved: a failure in one cannot reach the other.

That last property is the point. Work can be issued on one stream without one
failing case poisoning the rest, which is what makes the design usable from a
test — and it is why nothing here is scoped to a stream.

### What a flag means

Precisely: *these bytes have no writer*. Work that was going to fill them did
not, so reading them would hand back whatever was there before.

It is not "an error happened near here". A buffer a failed launch never
touched reads normally, and a buffer that is written again — by a relaunch or
a host copy — reads normally from then on, because it has a writer again.

## The four pieces

| | what it holds |
|---|---|
| [`Taint`](crates/cubecl-runtime/src/memory_management/taint.rs) | which byte ranges of one allocation are flagged, and by which failure |
| [`ErrorGraph`](crates/cubecl-runtime/src/memory_management/error_graph.rs) | the errors themselves, refcounted by the allocations carrying them, plus the skip records that make a report a path |
| [`Failures`](crates/cubecl-runtime/src/stream/failures.rs) | one store per device: the graph, and a pooled vector the scopes stage write sets in |
| [`ExecuteScope`](crates/cubecl-runtime/src/stream/execute_scope.rs) | the only way a backend touches any of it |

`Taint` holds ranges rather than a whole-allocation bit because the grain
matters: a host write covering one row of a tensor must not clear the flag on
all of it, and a launch that failed writing one region must not fail reads of
the rest. It stores one claim of one range inline, which is what a working
program's launches actually cost, and spills to the heap beyond that.

The refcount runs the other way from the obvious one: an error lives as long
as some allocation still carries it, so freeing the last buffer that names a
failure drops the failure. A program that manages its memory has managed its
error state for free.

## The scope

Every unit of device work runs inside an `ExecuteScope`, and which kind it is
is decided in the constructor:

```rust
ExecuteScope::launching(server, kernel, stream, reads, written)   // a kernel
ExecuteScope::over(server, stream, written)                       // everything else
    .execute(|server| { /* enqueue the work */ })
```

- **`over`** — work that reads nothing it must trust: a host copy, a graph
  replay, a launch that never compiled. It always **enters**: the write set is
  claimed by a provisional failure, and leaving either releases the claim or
  swaps the real error in.
- **`launching`** — a kernel. If an input carries a failure it **skips**: the
  write set is pointed at *that* failure rather than a new one, and the body
  never runs.

A scope is one or the other before it exists, which is not a convenience.
Entering on a skip would mint a provisional failure, have it overwritten by
the propagated one, and leave it in the graph forever — the exit that prunes
it is on the path that does not run.

The body is a closure and the scope is not an RAII guard. `#[must_use]` says
nothing about a bound value on a path that returns early, and `Drop` cannot
reach the failure store and must not assert during an unwind. A path that
never reaches the exit — a panic mid-launch above all — leaves the write set
claimed, which is exactly what a later read has to fail on.

### Skip, do not run

A launch whose input cannot be trusted is not merely wasted device time. A
buffer holding garbage can be read as a dynamic cube count or as gather
indices, dispatching an absurd grid or scattering into memory that carried no
failure at all. So the launch is skipped and its outputs take the failure,
exactly as a failed launch's would.

The skip is recorded as an edge — the kernel, the node it needed, the nodes it
produced — so a read two hops downstream reports the root cause *and* the path
to it.

### Where the sets come from

The read set and the write set are the compiled kernel's own answer — the
visibility analysis can prove a buffer write-only or dead, which nothing else
can. A kernel that never compiled kept no answer, so the launch site's
declaration stands in: the generated launch functions declare what each
signature proves — `&Tensor` cannot be written, `&mut Tensor` may be read —
and an argument that aliases another writes it in place, so its declaration
lands on the aliased buffer. A launch with neither answer over-names:
everything is checked and everything is claimed, because a spurious loud
failure is recoverable and a stale buffer reading clean is not.

The declaration is what keeps one candidate that fails to compile from
poisoning an autotune sweep: the failure claims the outputs the candidate
declared, never the inputs every other candidate still has to read.

### The scope is the capture window's only informant

On a stream recording a graph, the scope's exit is also what the window hears.
A clean exit hands it the write set — what the graph will write is what a
scope inside it wrote, recorded in one place so the two cannot disagree. A
failed or skipped exit dooms the window: the recording is missing an
operation, `end_capture` refuses to seal it, and the buffers the recorded
launches were given are claimed instead. A backend exposes its capture state
through one `WriteScoped::capturing` accessor and gets all of this without
wiring any of it; a server that captures nothing returns `None` and the
reports are no-ops.

The window owns what its recording points at, too: a host source recorded as
a memcpy node rides the window onto the graph rather than the drop queue,
because every replay reads through that raw pointer again. And its write set
is deduplicated by claim — the allocation *and* the byte range — because the
taint bookkeeping is range-exact, and two tensors carved from one batched
allocation are two claims, not one.

## What is deliberately absent

**No queue.** A failure is not owed to a later flush. It is on the buffers, and
a read of one of them is the report.

**No per-stream error state, with one exception.** A stream is not an edge, so
anything scoped to one breaks the isolation above. Metal keeps the exception:
a command-buffer fault arrives in a completion handler that can name no buffer,
so it is recorded on the stream and every later wait on that stream fails on
it. The consequence is the one this rule exists to prevent — after a Metal
fault, two workflows sharing that stream do contaminate each other — so on
Metal the isolation property holds for enqueue-time failures and not for
execution-time ones. The slot is sticky rather than report-once because
clearing it is exactly how stale bytes would start reading clean again.

**No answer for a failed allocation.** `initialize_memory` has no error
channel and nothing to taint — a reservation that never got its storage has no
binding for a claim to sit on — so every backend panics there. Device OOM is
the one failure this model cannot carry.

**No device-wide fault.** There was one, and it made the next flush or sync
fail whatever it was flushing, including work sharing no buffer with whatever
broke. It also protected nothing: the slot was taken — one report, then
cleared — after which the stale buffers read clean, never having been claimed.
Each site now claims what it can name and logs when it cannot.

## Assumptions

These are load-bearing and worth challenging.

**A claim covers enqueue, not execution.** `enter_write` claims, the body
submits, `exit_write` releases — and submission returns long before the kernel
runs. So a claim answers *"was this work accepted by the driver"*. An
asynchronous failure arrives after every related claim is gone.

**Asynchronous failures reach the caller by other means.** A read awaits a
fence and propagates its error; a faulted Metal command buffer records the
fault on its stream's sticky slot and forces its event, so every dependent
wait returns promptly and fails on the slot — sticky rather than
report-once, because clearing it is exactly how stale bytes would start
reading clean again. Paths that hand back device data *without* awaiting a
fence — `get_resource` returns a pointer — do not have this cover.

**A stream survives the errors it reports.** Measured for allocation failure
on gfx1151: the device serves the next request normally. **Not** measured for
an illegal address, which on some hardware poisons the whole context; the
experiment was skipped because provoking it means faulting the APU that drives
the display. If that assumption is wrong, the honest state after such an error
is every allocation on the device flagged, which nothing currently does.

**Some failures cannot be attributed.** A Metal command-buffer completion
handler knows the staging temporaries and the event, never the outputs — so
its fault claims no buffer and fails the whole stream's waits instead,
coarse by necessity.

## Where it is enforced

- `crates/cubecl-core/src/runtime_tests/stream_errors.rs` — thirteen properties
  every backend runs, plus one more for the backends that resolve a dynamic
  cube count on the host, including that two workflows interleaved on one stream do
  not contaminate each other, that a read reports the root cause two hops down,
  and that a rewrite makes a stale buffer readable again.
- `crates/cubecl-runtime/tests/taint_property.rs` — a randomised model check
  that a read returns bytes if and only if their last writer succeeded, plus
  the scope's own properties: a mid-launch panic leaves the set claimed, a
  partial host write releases only the bytes it covers, a skip mints no
  failure of its own.
- `crates/cubecl-runtime/src/memory_management/error_graph.rs` — unit tests for
  the refcount and the skip-path walk.

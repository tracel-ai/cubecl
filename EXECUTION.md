# The execution module: replacing dry run

Status: agreed direction, being implemented on `feat/build-progress`.
Replaces `cubecl_runtime::dry_run` with `cubecl_runtime::execution`.

## Why

"Dry run" names what is skipped, not what is produced.
Its whole purpose is side effects: compiled kernels, filled caches, tune results.
Under `CompileAndAutotune`, autotune launches really execute.

The current module also has a real bug.
Its narrow overrides (`RealRun`, `CompileOnly`) are `thread_local!`.
Under `StreamPolicy::PerTask` a tokio task hops threads across `.await` and loses them.

## Two levels: the process's policy, a stream's mode

**The policy is the whole process's.** It applies to every launch, on every device.
That is not a simplification, it is where the state lives:

- the compile queue belongs to each server (`cubecl-server/src/compiler/loader.rs`)
- the tune cache is keyed by tune key, not by stream
- tune measurements are skewed by any other work on the device
- memory pools are per device and share pages across streams (`PageMapping::current`)

A per-stream policy would only be true for dropping a launch.
Compiling, tuning and memory would leak between streams that disagree.
Per-stream policies wait until a caller needs one and those four have an answer.

**A stream only knows whether its launches run.** Its mode is `Execute` or `Compile`.
It does not know why: autotune is a caller on top of it, not a state inside it.
The policy sets every stream's default mode, and decides what `Compile` means for the pass:
queue the kernel for a batch, or compile it now.
A measurement — autotune's candidates, a throughput probe — switches the stream it measures on to `Execute`
while it measures, and hands it back.

## Naming

`Policy` for behavior that is set and resolved later, `Mode` for a stream's one bit,
and value enums describing the math keep their names (`QuantMode`, `ClampMode`).
The module is `execution`: it holds the policy, the stream modes, and the statistics of what they triggered.
`runtime` is taken by `cubecl-runtime/src/runtime.rs`, and `global` names a scope rather than a concept.
The module docs state the scope of each level, so type names do not repeat it.

## Types

```rust
// cubecl_runtime::execution

/// What the process does with the work it is asked to run.
pub enum ExecutionPolicy {
    /// Launches run. What the process does when no override is open.
    Execute,
    /// Launches and tune candidates queue their kernels and are dropped.
    /// The queue compiles in one batch, at the next launch that loads a kernel.
    /// A tune measures and decides nothing.
    CompileOnly,
    /// Launches compile their kernels and are dropped.
    /// A tune measures its candidates for real.
    CompileAndAutotune,
}

/// Applies a policy to the whole process while it lives, counting what it
/// triggers into a collector; restored on drop.
pub struct ExecutionOverride { /* private */ }

impl ExecutionOverride {
    pub fn new(policy: ExecutionPolicy, collector: &StatisticsCollector) -> Self;
}

/// Whether a stream's launches run.
pub enum StreamMode {
    Execute,
    /// Compile the kernel — now, or queued, as the policy says — and drop the launch.
    Compile,
}

/// Sets the mode of one client's stream while it lives, restored on drop.
/// Replaces `RealRun` and the `CompileOnly` guard.
pub struct StreamModeOverride { /* private */ }

impl StreamModeOverride {
    pub fn new(mode: StreamMode, client: &Client) -> Self;
}

/// What a server does with one launch: the resolved verdict.
pub enum LaunchAction {
    Execute, // compile if needed, then run
    Compile, // compile if needed, cache, drop the launch
    Queue,   // queue the kernel for a batch compile, drop the launch
}
```

Resolution for a launch, on the stream it goes out on:

```
mode   = the newest live StreamModeOverride on the launch's stream
       → the policy's default: Execute for Execute, Compile otherwise

action = (Execute, _)                  → Execute
         (Compile, CompileOnly)        → Queue
         (Compile, CompileAndAutotune) → Compile
         (Compile, Execute)            → Queue: one stream gathering while the process runs
```

`StreamModeOverride` keys on the client's stream, `client.stream_id()`: the stream the launches actually use.
A client bound to an explicit stream does not follow `StreamId::current()`,
and a tokio task keeps its stream across `.await` under `PerTask`.
Guards can drop out of order, so each holds an id and the process keeps the set of live ones;
a launch checks one atomic count before it looks, so no live guard costs nothing.

Overrides follow today's rules: overlapping overrides of one policy and one collector compose, anything else panics.

## Statistics

```rust
// cubecl_runtime::execution (statistics)

/// Collects what the overrides opened with it triggered.
/// Not `Clone`: whoever holds it opens overrides. A reader holds a `StatisticsReader`.
pub struct StatisticsCollector { /* private, holds its id */ }

impl StatisticsCollector {
    pub fn new() -> Self;
    pub fn reader(&self) -> StatisticsReader;
    pub fn statistics(&self) -> ExecutionStatistics;
}

/// Reads a collector's statistics from any thread, and opens nothing.
#[derive(Clone)]
pub struct StatisticsReader { /* private */ }

/// A snapshot. Every count only grows.
pub struct ExecutionStatistics {
    pub compilation: CompilationStatistics,
    pub autotune: AutotuneStatistics,
}

/// The kernels the launches under an override obtained.
pub struct CompilationStatistics {
    /// Queued, or asked for by a launch that found them neither loaded nor queued.
    /// A kernel tried again after failing registers again.
    pub registered: usize,
    /// Compiled by the backend's compiler.
    pub compiled: usize,
    /// Read from the compilation store instead.
    pub loaded: usize,
    /// Failed to compile.
    pub failed: usize,
}

/// The autotune keys tuned under an override.
/// A key with one candidate is answered, not tuned, and counts nowhere.
pub struct AutotuneStatistics {
    /// Gathered by a `CompileOnly` pass, or reached ungathered by a tune that measures.
    pub registered: usize,
    /// Tuned by measuring their candidates.
    pub measured: usize,
    /// Every candidate failed: the pick was made unmeasured.
    pub failed: usize,
}
```

Each type has `settled()`, the outcomes added up, never more than `registered`:
what a frontend draws a bar from.
The counting itself exists once, private: a count is registered before its outcome,
and a snapshot reads the outcomes before the registrations.

The rules:

- A kernel counts to the collector of the override that registered it, whenever and wherever its batch compiles.
- A kernel waiting on another kernel of its batch with the same source settles when that source is done.
- A tune counts to the collector of the override that registered it, even when its pick commits with no override open.
- A key reached inside another key's candidates, under `CompileOnly`, is not registered:
  the `CompileAndAutotune` pass stops a plan at its first close-enough candidate and may never reach it.
  It registers if it is ever measured, as an ungathered key does.

`Progress` is avoided because metabolic owns that word.
`Observation` overlaps `logging::observer`, `Monitoring` is the `cubecl-monitoring` crate.

Usage across two passes:

```rust
let collector = StatisticsCollector::new();
let reader = collector.reader(); // for another thread
{
    let _o = ExecutionOverride::new(ExecutionPolicy::CompileOnly, &collector);
    model.forward(input);
}
{
    let _o = ExecutionOverride::new(ExecutionPolicy::CompileAndAutotune, &collector);
    model.forward(input);
}
let stats = collector.statistics();
```

An override of `Execute` with a collector counts what a real run compiles and tunes:
what a test asserting that a warmed-up run compiles nothing reads.

## Mapping from today

| Today | New |
|---|---|
| `dry_run` module | `execution` module |
| `DryRun` | `StatisticsCollector` |
| `DryRunId` | private to the collector |
| `DryRun::pass(scope)` / `DryRunPass` | `ExecutionOverride::new(policy, &collector)` |
| `DryRunScope::Compile` / `Profile` | `ExecutionPolicy::CompileOnly` / `CompileAndAutotune` |
| `DryRunObserver` | `StatisticsReader` |
| `DryRunObservation` | `ExecutionStatistics` |
| `Progress { requested, settled }` | `CompilationStatistics` / `AutotuneStatistics` |
| `DryRunCounter`, `counted()` | `StatisticsRecorder`, `execution::recorder()`: for the loader and the tuner |
| `LaunchMode::{Execute, Skip, CompileOnly}` | `LaunchAction::{Execute, Compile, Queue}` |
| `RealRun` | `StreamModeOverride::new(StreamMode::Execute, &client)` |
| `CompileOnly` guard | `StreamModeOverride::new(StreamMode::Compile, &client)` |
| `launch_mode()` | `execution::launch_action(stream)` |
| `dry_run()`, `dry_run_scope()` | `execution::policy() -> ExecutionPolicy` |

## Follow-ups, each its own change

- `ExecutionMode` (cubecl-ir) becomes `BoundsCheck`.
  It decides bounds checking, compiled into the kernel, and has nothing to do with execution.
  It derives `Serialize` and is part of `KernelId`, so keep the variant spellings.
  It reaches cubek and burn, so it moves the rev chain on its own.
- `BoundsCheckMode` (config) becomes `BoundsCheckPolicy`. The config key `check_mode` keeps working.
- `TuneRecord::dry_run: bool` becomes the policy the tune ran under.
  cubecl-inspect in metabolic reads it, so migrate both sides together.
  Until then it reads `true` for any policy but `Execute`.
- `StreamId { pub value: u64 }`: make the field private.
  `StreamModeOverride` keys on stream ids, and a hand-built id colliding with an allocated one
  would apply a mode to the wrong work.

## Out of scope

- Per-stream and per-device policies.
- `StreamPolicy` keeps its name, nothing collides with it now.

## Open questions

- Graph capture under an override. A capture under `CompileOnly` or `CompileAndAutotune` records
  launches that were dropped, and a replay of it is wrong without saying so — but metabolic's captured
  transformer captures during its build on purpose, its step switched to `Execute`
  (`metabolic-models/src/transformer/captured.rs`). Refusing a capture whose stream is not in
  `Execute` mode keeps that case and closes the silent one.

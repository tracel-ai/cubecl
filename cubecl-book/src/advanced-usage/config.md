# Configuration

CubeCL provides a flexible and powerful configuration system to control logging, autotuning,
profiling, and compilation behaviors.

## Overview

By default, CubeCL loads its configuration from a TOML file (`cubecl.toml` or `CubeCL.toml`) located
in your current directory or any parent directory. If no configuration file is found, CubeCL falls
back to sensible defaults.

You can also override configuration options using environment variables, which is useful for CI,
debugging, or deployment scenarios.

## Configuration File Structure

A typical `cubecl.toml` file might look like this:

```toml
[profiling]
logger = { level = "basic", stdout = true }

[autotune]
level = "balanced"
logger = { level = "minimal", stdout = true }

[compilation]
logger = { level = "basic", file = "cubecl.log", append = true }
```

Each section configures a different aspect of CubeCL:

- **profiling**: Controls performance profiling and logging.
- **autotune**: Configures the autotuning system, which benchmarks and selects optimal kernel
  parameters.
- **compilation**: Manages kernel compilation logging and cache.

## Configuration Options

### Profiling

The `[profiling]` section controls how CubeCL logs profiling information.

**Log Levels:**

- `disabled`: No profiling logs.
- `minimal`: Only logs which kernels run.
- `basic`: Adds basic profiling info.
- `medium`: More detailed profiling.
- `full`: Maximum detail.

**Example:**

```toml
[profiling]
logger = { level = "basic", stdout = true }
```

### Autotune

The `[autotune]` section configures how aggressively CubeCL autotunes kernels and where it stores
autotune results.

**Autotune Levels:**

- `minimal`: Fastest, least thorough.
- `balanced`: Good trade-off (default).
- `extensive`: More thorough.
- `full`: Most thorough, slowest.

**Log Levels:**

- `disabled`, `minimal`, `full`

**Example:**

```toml
[autotune]
level = "balanced"
logger = { level = "minimal", stdout = true }
```

**Cache Location (if enabled):**

- `local`: Current directory
- `target`: Project's `target` directory (default)
- `global`: System config directory
- `file`: Custom path

### Compilation

The `[compilation]` section manages logging and caching for kernel compilation.

**Log Levels:**

- `disabled`: No logs.
- `basic`: Logs when kernels are compiled.
- `full`: Logs full details, including source code.

**f16 evaluation** (`f16_evaluation`, CPU runtime only): how far an f16 intermediate is carried in
f32 before it is rounded back. Unset, it is `per-operation` on a host with f16 arithmetic of its
own and `chain` elsewhere.

- `per-operation`: round after every operation, as a GPU does.
- `chain`: round where a value is stored or read by anything but arithmetic.
- `accumulators`: also hold a private f16 variable in f32 where that removes converts.

**Example:**

```toml
[compilation]
logger = { level = "basic", file = "cubecl.log", append = true }
f16_evaluation = "chain"
```

### Streaming

The `[streaming]` section manages logging and stream configurations.

**Log Levels:**

- `disabled`: No logs.
- `basic`: Basic streaming information is logged such as when streams are merged.
- `full`: Full streaming details are logged.

**Example:**

```toml
[streaming]
logger = { level = "basic", file = "cubecl.log", append = true }
max_streams: 4
```

### Memory

The `[memory]` section controls memory-related logging and the persistent-memory policy.

**Log Levels:**

- `disabled`: No logs.
- `basic`: Basic memory events, such as creating memory pages and manual cleanups.
- `full`: Detailed memory information.

**Persistent memory** (`persistent_memory`): controls the pool used for long-lived allocations
such as model weights.

- `enabled` (default): used only when explicitly requested (e.g.
  `Client::memory_persistent_allocation`).
- `size-match`: like `enabled`, and automatic allocations whose size matches an existing
  persistent bucket are served from the persistent pool too.
  A good fit for training, less so for inference.
- `disabled`: requests to switch to persistent allocation are ignored.
- `enforced`: every allocation is persistent. May cause out-of-memory errors when tensor sizes
  vary.

**Example:**

```toml
[memory]
logger = { level = "basic", stdout = true }
persistent_memory = "enabled"
```

There is no setting for the memory pools.
The memory management lays them out from the allocations it serves, see
[Memory pools](#memory-pools) below.
A leftover `pools` entry in `[memory]` is a load error.

## Environment Variable Overrides

CubeCL supports several environment variables to override configuration at runtime:

- `CUBECL_DEBUG_LOG`: Controls logging output.
  - `"stdout"`: Log to stdout.
  - `"stderr"`: Log to stderr.
  - `"1"`/`"true"`: Log to `/tmp/cubecl.log`.
  - `"0"`/`"false"`: Disable logging.
  - Any other value: Treated as a file path.
- `CUBECL_DEBUG_OPTION`: Sets log verbosity.
  - `"debug"`: Full compilation and autotune logs, medium profiling.
  - `"debug-full"`: Full logs for all.
  - `"profile"`, `"profile-medium"`, `"profile-full"`: Set profiling log level.
- `CUBECL_AUTOTUNE_LEVEL`: Sets autotune level.
  - `"minimal"`/`"0"`
  - `"balanced"`/`"1"`
  - `"extensive"`/`"2"`
  - `"full"`/`"3"`
- `CUBECL_CPU_F16_EVAL`: Sets `compilation.f16_evaluation`.
  - `"per-operation"`, `"chain"`, `"accumulators"`

**Example (Linux/macOS):**

```sh
export CUBECL_DEBUG_LOG=stdout
export CUBECL_AUTOTUNE_LEVEL=full
```

## Programmatic Configuration

You can also set the global configuration from Rust code before CubeCL is initialized:

```rust
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

// Seed from `cubecl.toml` and `CUBECL_*` env vars, then override in code.
let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
config.autotune.level = cubecl::config::autotune::AutotuneLevel::Extensive;
CubeClRuntimeConfig::set(config);
```

> **Note:** You must call `CubeClRuntimeConfig::set` before any CubeCL operations, and only once
> per process.

Two sharp edges:

- `set` panics if the configuration was already loaded.
  It must run before any CubeCL call that touches a client, autotune, or logging.
- Seeding with `from_current_dir().override_from_env()` as above keeps `cubecl.toml` and
  `CUBECL_*` env vars in effect.
  Starting from `CubeClRuntimeConfig::default()` discards them.

## Memory pools

The pools of a runtime's main GPU memory lay themselves out.
Nothing is measured or configured per workload.

- Small allocations share a sliced pool of their own.
- Everything else is carved from pages sized to the largest allocation served so far.
- When an allocation outgrows the pages, pages of the new size take over.
  The old ones are returned as they empty.
- When memory runs low, or too many page sizes pile up, what lives on the old pages is relocated
  onto the new ones so they can be returned sooner.
- `memory_cleanup` relocates first, then returns every page nothing uses.

A device that cannot sub-slice a page, and every wasm target, gets one page per allocation in
size buckets instead.

To see what the pools hold, ask the client for a report:

```rust
use cubecl::MemoryScope;

let report = client.memory_report(MemoryScope::Device);
println!("{}", report.usage());
for stream in &report.streams {
    for pool in &stream.pools.dynamic {
        println!("{:?}: {} pages", pool.kind, pool.pages);
    }
}
```

`MemoryScope::CurrentStream` limits the report to the stream the client issues on.

## Logging

CubeCL supports logging to multiple destinations simultaneously:

- File (with append/overwrite)
- Stdout
- Stderr
- Rust `log` crate (for integration with other logging frameworks)

You can configure these in the `logger` field for each section.

## Saving the Default Configuration

To generate a default configuration file:

```rust
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

CubeClRuntimeConfig::save_default("cubecl.toml").unwrap();
```

## Example: Full Configuration

```toml
[profiling]
logger = { level = "medium", stdout = true }

[autotune]
level = "extensive"
logger = { level = "full", file = "autotune.log", append = false }

[compilation]
logger = { level = "full", file = "compile.log", append = true }
```

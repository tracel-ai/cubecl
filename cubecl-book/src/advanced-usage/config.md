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

**Example:**

```toml
[compilation]
logger = { level = "basic", file = "cubecl.log", append = true }
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
  `ComputeClient::memory_persistent_allocation`).
- `disabled`: requests to switch to persistent allocation are ignored.
- `enforced`: every allocation is persistent. May cause out-of-memory errors when tensor sizes
  vary.

**Example:**

```toml
[memory]
logger = { level = "basic", stdout = true }
persistent_memory = "enabled"
```

**Memory pools are a programmatic setting, not a config-file one.** Pool layouts are dynamic.
They are sized at runtime from the workload (model size, sequence length, batch) and change
between workloads, so they must not freeze at process startup. See
[Memory pool layouts](#memory-pool-layouts) below. A leftover `pools` entry in `[memory]` is a
load error.

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

## Memory pool layouts

The pool layout of a runtime's **main GPU** memory is configured at runtime through the compute
client, never through a config file, so it can change between workloads instead of freezing at
startup:

```rust
use cubecl::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl::config::size::MemorySize;

let applied = client.configure_memory_pools(&MemoryPoolsConfig::Explicit(vec![
    MemoryPoolConfig::Sliced {
        page_size: MemorySize(page_bytes),
        max_slice_size: None,
        max_pool_size: Some(MemorySize(pages * page_bytes)),
        dealloc_period: None,
    },
]));
assert!(applied, "something was still live in the old pools");
```

The value is either a preset (`MemoryPoolsConfig::Preset` with `SubSlices` or `ExclusivePages`,
matching the runtime defaults) or an explicit pool list. At allocation time, the first pool that
accepts an allocation's size serves it. Auxiliary pools (pinned CPU, staging, uniforms) and the
persistent pool are never affected.

Semantics:

- The **calling stream's** pools are rebuilt in place, provided nothing is live in them.
  Reconfigure at a quiescent point, for example right after unloading a model and running
  `memory_cleanup`. When something is still live, the call returns `false`, the old layout is
  kept, and a memory log line says so. Garbage-collection tasks release their pins
  asynchronously, so a `false` right after a cleanup usually succeeds on a retry.
- Every stream **created afterwards** is built with the new layout, so workloads with different
  layouts can coexist on different streams.
- An invalid layout (empty list, too many pools, zero `page_size`, `max_slice_size` larger than
  `page_size`, `max_pool_size` smaller than `page_size`) panics with a descriptive message. An
  explicit layout is never silently replaced.
- `page_size` may exceed the device's reported `max_page_size`: that value is a sizing heuristic
  for the default layouts, not an allocation limit. A page the device truly cannot allocate
  fails at allocation time.
- A hard-capped sliced arena gives a **fixed memory footprint**: allocations of every size reuse
  the same pages, and when the cap is reached and nothing fits after coalescing, the allocation
  fails with a pool-capacity error instead of growing silently.

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

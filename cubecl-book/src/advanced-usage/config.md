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

The `[memory]` section controls memory-related logging and the memory pools of every runtime.

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

**Memory pools** (`pools`): overrides the pool layout of every runtime's **main GPU** memory.
Omit it to keep each runtime's own default. Auxiliary pools (pinned CPU, staging, uniforms) are
never affected. The value is either a preset name:

```toml
[memory]
pools = "sub-slices" # or "exclusive-pages"
```

or an explicit pool list. At allocation time, the first pool that accepts an allocation's size
serves it. Sizes accept raw integers (bytes) or human-readable strings with binary units
(`"8KiB"`, `"512MB"`, `"20GiB"`).

```toml
[memory]
[[memory.pools]]
type = "exclusive"       # one page per allocation
max_alloc_size = "8KiB"  # small buffers (e.g. kernel metadata)
dealloc_period = 10000   # deallocate unused pages periodically (omit to keep them forever)

[[memory.pools]]
type = "sliced"          # allocations are slices of large pages
page_size = "20GiB"
max_slice_size = "20GiB" # optional, defaults to page_size
max_pool_size = "20GiB"  # hard cap: exceeding it is an error, never silent growth
```

A single sliced arena with a hard cap, as above, gives a **fixed memory footprint**: allocations
of every size reuse the same pages instead of each size-bucketed pool retaining its own peak
reservation — useful when the maximum working set is known up front, such as an LLM inference
server whose KV-cache size is fixed. When the cap is reached and nothing fits after coalescing,
the allocation fails with a pool-capacity error instead of growing.

Notes:

- An invalid layout (empty list, zero `page_size`, `max_slice_size` larger than `page_size`,
  `max_pool_size` smaller than `page_size`) panics at client creation with a descriptive
  message — an explicit memory override is never silently replaced.
- `page_size` may exceed the device's reported `max_page_size`: that value is a sizing heuristic
  for the default layouts, not an allocation limit. A page the device truly cannot allocate
  fails at allocation time.
- Runtimes that create one memory management per stream (CUDA, HIP) apply the layout — and any
  `max_pool_size` cap — per stream.

**Example:**

```toml
[memory]
logger = { level = "basic", stdout = true }
persistent_memory = "enabled"
pools = "sub-slices"
```

Since pool sizes are often computed at runtime (model size, sequence length, batch), the
`pools` setting is commonly set programmatically instead of in a file — see
[Programmatic Configuration](#programmatic-configuration).

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

let mut config = CubeClRuntimeConfig::default();
config.autotune.level = cubecl::config::autotune::AutotuneLevel::Extensive;
CubeClRuntimeConfig::set(config);
```

> **Note:** You must call `CubeClRuntimeConfig::set` before any CubeCL operations, and only once
> per process.

This is the recommended way to configure memory pools whose sizes are computed at runtime, such
as a fixed arena covering an LLM KV cache plus activations:

```rust
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::config::memory::{MemoryPoolConfig, MemoryPoolsConfig};
use cubecl::config::size::MemorySize;

// Start from the file/env configuration so `set` doesn't discard it.
let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
config.memory.pools = Some(MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
    page_size: MemorySize(budget_bytes),
    max_slice_size: None,
    max_pool_size: Some(MemorySize(budget_bytes)),
    dealloc_period: None,
}]));
CubeClRuntimeConfig::set(config);
// ... only now create the first client/device.
```

Two sharp edges:

- `set` panics if the configuration was already loaded — it must run before *any* CubeCL call
  that touches a client, autotune, or logging.
- `set` bypasses `cubecl.toml` and `CUBECL_*` env vars unless you seed the value with
  `from_current_dir().override_from_env()` as above.

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

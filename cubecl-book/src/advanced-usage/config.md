# Configuration

CubeCL provides a flexible and powerful configuration system to control logging, autotuning, profiling, and compilation behaviors.

## Overview

By default, CubeCL loads its configuration from a TOML file (`cubecl.toml` or `CubeCL.toml`) located in your current directory or any parent directory. If no configuration file is found, CubeCL falls back to sensible defaults.

You can also override configuration options using environment variables, which is useful for CI, debugging, or deployment scenarios.

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
- **autotune**: Configures the autotuning system, which benchmarks and selects optimal kernel parameters.
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

The `[autotune]` section configures how aggressively CubeCL autotunes kernels and where it stores autotune results.

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
let config = cubecl::config::GlobalConfig {
    profiling: ...,
    autotune: ...,
    compilation: ...,
};
cubecl::config::GlobalConfig::set(config);
```

> **Note:** You must call `GlobalConfig::set` before any CubeCL operations, and only once per process.

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
cubecl::config::GlobalConfig::save_default("cubecl.toml").unwrap();
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

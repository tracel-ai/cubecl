# Constants

CubeCL gives information about the runtime position and certain properties via Rust constants,
similar to CUDA (i.e. `threadIdx`) or GLSL (i.e. `gl_LocalInvocationID`). These will be translated
to target-specific intrinsics at `comptime`. Please note that _all builtin constants show up as `2`
in `rust-analyzer`_. This is because they're not real Rust constants and are just tokens that are
later substituted. Setting them to `2` prevents most (though not all) clippy issues around
divide-by-zero, underflow, and redundant expressions. If you still see any errors, it's safe to
ignore them (via `#[allow(clippy::whatever_lint)]`).

## Plane

GPUs are organized into groups of threads that operate in lockstep. This is similar to but not quite
the same as SIMD in CPUs. In CubeCL, these units are called `plane`s. If you're familiar with CUDA,
they are `warp`s, in Vulkan they are `subgroup`s, in Metal they are `SIMD group`s.

CubeCL offers several builtins to deal with `plane`s:

#### `PLANE_DIM`

Each plane's dimensions at runtime. Generally, supported plane sizes can be queried in the hardware
properties. However, not all hardware and backends use a fixed size for this, and they can vary
based on several factors (i.e. register pressure on certain Intel GPUs). The builtin gives you the
_actual size_ that was picked by the driver.

#### `PLANE_POS`

The position of the plane within the `cube`. This is not meaningful in terms of hardware except in
rare cases, but makes it easy to divide work between multiple `plane`s.

#### `UNIT_POS_PLANE`

The position of a unit inside the plane it's in, from 0 to `PLANE_DIM`. This is mainly useful when
doing cooperative `plane` ops, like `plane_broadcast` and `plane_shuffle` to determine the relative
position of the current unit.

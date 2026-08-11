use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::MemoryAccess;

use crate::throughput::LaunchConfig;

/// The buffer geometry and launch shape a memory probe uses to move a working
/// set of a given size in one pass.
///
/// Shared by the copy and read probes: both walk one line per thread per step
/// over buffers of identical size, and both have to shrink the launch rather
/// than the working set when asked for a small one.
#[derive(Clone, Copy, Debug)]
pub struct MemoryProbe {
    /// Lines in each buffer.
    pub num_lines: usize,
    /// Bytes in each buffer.
    pub buffer_bytes: usize,
    /// Cubes to dispatch, which is [`LaunchConfig::cube_count`] unless the
    /// working set is too small to give every thread a line.
    pub cube_count: usize,
}

impl MemoryProbe {
    /// Sizes a probe moving `working_set` bytes per pass, split evenly across
    /// the buffers `access` touches, and clamped to the device's maximum
    /// allocation.
    ///
    /// The launch shrinks with the working set instead of the working set
    /// growing to fill the launch. A small buffer measured with the full
    /// dispatch would either read the same line from many threads or be padded
    /// back up to a large buffer, and neither is the small-kernel behaviour the
    /// curve exists to describe: a kernel that moves little has little in
    /// flight, and that is precisely what limits it.
    ///
    /// Both kernels index with `len - ABSOLUTE_POS`, so a thread past the end
    /// of the buffer would underflow into an enormous loop count. `num_lines`
    /// is therefore never below the threads actually dispatched.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        config: LaunchConfig,
        line_bytes: usize,
        access: MemoryAccess,
        working_set: usize,
    ) -> Self {
        let max_alloc = client.properties().memory.max_page_size as usize;
        let buffers = access.buffers() as usize;
        let target = (working_set / buffers).min(max_alloc);

        let num_lines = (target / line_bytes).max(1);
        let cube_count = (num_lines / config.cube_dim).clamp(1, config.cube_count);
        // At most one cube's worth of padding, and only for a working set far
        // below the smallest the sweep asks for.
        let num_lines = num_lines.max(cube_count * config.cube_dim);

        Self {
            num_lines,
            buffer_bytes: num_lines * line_bytes,
            cube_count,
        }
    }
}

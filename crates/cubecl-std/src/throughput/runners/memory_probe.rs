use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::ElemType};
use cubecl_runtime::{
    server::Handle,
    throughput::{DEFAULT_BUFFER_BYTES, MemoryAccess},
};

use crate::throughput::LaunchConfig;

/// The buffer geometry and launch shape a memory probe uses to move a working
/// set of a given size in one pass.
///
/// Shared by the copy and read probes: both walk one line per thread per step
/// over buffers of identical size, and both measure a small working set by
/// moving a small *window* over a large buffer rather than by allocating a
/// small buffer.
///
/// The window is what makes a small working set mean anything. A small buffer
/// read over and over stays in cache after the first pass, so the probe would
/// report cache bandwidth for every size below the cache and the curve would
/// describe residency instead of size. Reading a fresh window each pass keeps
/// the data cold — by the time the window comes back around, a whole buffer of
/// traffic has evicted it — so what varies across the sweep is how much a pass
/// moves, which is the thing being measured.
#[derive(Clone, Copy, Debug)]
pub struct MemoryProbe {
    /// Lines in each buffer: as much as the device will allocate, whatever the
    /// working set, so a window has somewhere cold to come back to.
    pub pool_lines: usize,
    /// Lines one pass moves through, per buffer. Never more than
    /// [`pool_lines`](Self::pool_lines); equal to it at the top of the sweep,
    /// where the working set is the whole buffer.
    pub window_lines: usize,
    /// Bytes in each buffer.
    pub buffer_bytes: usize,
    /// Cubes to dispatch, which is [`LaunchConfig::cube_count`] unless the
    /// window is too small to give every thread a line.
    pub cube_count: usize,
    /// Whether the probe kernels address lines as per-thread contiguous runs
    /// rather than coalesced across threads. Set on plane-1 runtimes, where a
    /// worker has no plane neighbour to coalesce with.
    pub blocked: bool,
}

impl MemoryProbe {
    /// Sizes a probe moving `working_set` bytes per pass, split evenly across
    /// the buffers `access` touches.
    ///
    /// A blocked probe pins the launch to one cube: on the backend that
    /// addressing is for, a cube position is a loop wrapped around the whole
    /// kernel body, `n_iter` included, so more than one turns the window
    /// rotation between passes into a replay of the same narrow,
    /// blocked-addressed slice instead of a walk across the buffer, and cache
    /// serves it. A coalesced launch spreads one cube position's addresses
    /// across the full window regardless of cube count, so it has no such
    /// limit.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        config: LaunchConfig,
        line_bytes: usize,
        access: MemoryAccess,
        working_set: usize,
    ) -> Self {
        let max_alloc = client.properties().memory.max_page_size as usize;
        let blocked = config.plane_size == 1;
        let cube_count = if blocked { 1 } else { config.cube_count };

        Self::sized(
            max_alloc,
            config.cube_dim,
            cube_count,
            line_bytes,
            access,
            working_set,
            blocked,
        )
    }

    /// The geometry itself, with the device reduced to its allocation limit and
    /// launch shape so the sizing can be exercised without one.
    ///
    /// The launch shrinks with the window instead of the window growing to fill
    /// the launch. A small window measured with the full dispatch would either
    /// hand many threads the same line or be padded back up to a large one, and
    /// neither is the small-kernel behaviour the curve exists to describe: a
    /// kernel that moves little has little in flight, and that is precisely
    /// what limits it.
    fn sized(
        max_alloc: usize,
        cube_dim: usize,
        cube_count: usize,
        line_bytes: usize,
        access: MemoryAccess,
        working_set: usize,
        blocked: bool,
    ) -> Self {
        let buffers = access.buffers() as usize;

        // As large as allowed regardless of the working set: the pool is what
        // the window is cold against.
        let pool_bytes = max_alloc.min(DEFAULT_BUFFER_BYTES as usize);
        let pool_lines = (pool_bytes / line_bytes).max(1);

        let window_bytes = working_set / buffers;
        let window_lines = (window_bytes / line_bytes).max(1).min(pool_lines);

        let cube_count = (window_lines / cube_dim).clamp(1, cube_count);

        Self {
            pool_lines,
            window_lines,
            buffer_bytes: pool_lines * line_bytes,
            cube_count,
            blocked,
        }
    }
}

/// Writes every line of `handle`, once, before it is handed to a probe that
/// only reads it.
///
/// A fresh allocation is backed by the same physical zero page until its
/// first write, so every unwritten line a read-only probe visits is served
/// from that one cached page rather than from DRAM, inflating its reported
/// bandwidth well past the device's real ceiling. Writing real data in first
/// gives each line its own page, the way a buffer a real kernel reads
/// already got one from whoever produced it.
pub fn prime<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    pool_lines: usize,
    config: LaunchConfig,
    dtype: ElemType,
) {
    unsafe {
        prime_buffer::launch_unchecked(
            client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new(client, config.cube_dim),
            config.vector_size,
            BufferArg::from_raw_parts(handle.clone(), pool_lines),
            pool_lines,
            dtype,
        );
    }
    let _ = cubecl_core::future::block_on(client.sync());
}

#[cube(launch_unchecked)]
fn prime_buffer<I: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    len: usize,
    #[define(I)] _dtype: ElemType,
) {
    let stride = CUBE_DIM as usize * CUBE_COUNT;
    let steps = len.div_ceil(stride).max(1);

    for step in 0..steps {
        let idx = ABSOLUTE_POS + step * stride;
        if idx < len {
            output[idx] = Vector::<I, N>::empty();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;

    fn probe(working_set: usize, access: MemoryAccess) -> MemoryProbe {
        // 16-byte lines and 256-thread cubes, as a device with `vec4` f32 and a
        // full dispatch of 2048 cubes reports.
        MemoryProbe::sized(512 * MB, 256, 2048, 16, access, working_set, false)
    }

    #[test]
    fn the_pool_stays_large_however_small_the_window() {
        // 256 KiB of traffic, but still half a gigabyte to be cold against.
        let small = probe(256 * KB, MemoryAccess::Read);
        assert_eq!(small.buffer_bytes, 512 * MB);
        assert_eq!(small.window_lines, 256 * KB / 16);

        // A copy splits its working set across two buffers, so the same traffic
        // is half the window per buffer.
        let copy = probe(256 * KB, MemoryAccess::Copy);
        assert_eq!(copy.buffer_bytes, 512 * MB);
        assert_eq!(copy.window_lines, 128 * KB / 16);
    }

    #[test]
    fn the_window_fills_the_pool_at_the_top_of_the_sweep() {
        // The default working set is the whole buffer, where the window has
        // nowhere to move and the probe is the single-size one.
        let read = probe(512 * MB, MemoryAccess::Read);
        assert_eq!(read.window_lines, read.pool_lines);

        // And a window can never exceed the pool, whatever it is asked for.
        let huge = probe(8 * 1024 * MB, MemoryAccess::Read);
        assert_eq!(huge.window_lines, huge.pool_lines);
    }

    #[test]
    fn the_launch_shrinks_with_the_window() {
        // 16384 lines over 256-thread cubes is 64 cubes, not the full 2048:
        // a kernel this small has that little in flight, and that is the point.
        assert_eq!(probe(256 * KB, MemoryAccess::Read).cube_count, 64);

        // A window bigger than the dispatch keeps the full launch.
        assert_eq!(probe(512 * MB, MemoryAccess::Read).cube_count, 2048);

        // And a window smaller than a single cube still dispatches one, so the
        // kernel always has a thread to run.
        assert_eq!(probe(64, MemoryAccess::Read).cube_count, 1);
    }

    #[test]
    fn a_device_that_allocates_little_shrinks_the_pool_with_it() {
        // The pool is the allocation limit, and the window follows it down
        // rather than asking for memory the device does not have.
        let probe = MemoryProbe::sized(4 * MB, 256, 2048, 16, MemoryAccess::Read, 512 * MB, false);

        assert_eq!(probe.buffer_bytes, 4 * MB);
        assert_eq!(probe.window_lines, probe.pool_lines);
    }
}

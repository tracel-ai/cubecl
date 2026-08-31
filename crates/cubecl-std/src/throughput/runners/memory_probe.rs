use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::ElemType};
use cubecl_runtime::{
    server::Handle,
    throughput::{DEFAULT_BUFFER_BYTES, MemorySpec},
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
    /// Lines of each buffer the probe walks: what the window is cold against,
    /// and the whole buffer until the window spans several last level caches
    /// and is cold on its own.
    pub pool_lines: usize,
    /// Lines one pass moves through, per buffer. Never more than
    /// [`pool_lines`](Self::pool_lines); equal to it at the top of the sweep,
    /// where the window is its own pool.
    pub window_lines: usize,
    /// Bytes in each buffer, which is exactly the pool. The probe kernels take
    /// their wrap-around from `len()`, so a buffer longer than the pool would
    /// walk them past what [`prime`] wrote.
    pub buffer_bytes: usize,
    /// Cubes to dispatch, which is [`LaunchConfig::cube_count`] unless the
    /// window is too small to give every thread a line.
    pub cube_count: usize,
    /// Whether the probe kernels address lines as per-thread contiguous runs
    /// rather than coalesced across threads. Set on plane-1 runtimes, where a
    /// worker has no plane neighbour to coalesce with.
    pub blocked: bool,
}

/// All [`MemoryProbe::sized`] needs of a device, so the sizing can be exercised
/// without one.
#[derive(Clone, Copy)]
struct DeviceShape {
    /// Bytes the device will hand out in a single allocation.
    max_alloc: usize,
    /// Threads per cube, and cubes, of the probe's launch.
    cube_dim: usize,
    cube_count: usize,
}

impl MemoryProbe {
    /// Passes a launch carries for the window to come back to bytes a whole
    /// pool of traffic has since evicted, which is what keeps a window smaller
    /// than the cache cold.
    pub fn min_iterations(&self) -> usize {
        self.pool_lines.div_ceil(self.window_lines)
    }

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
        spec: MemorySpec,
    ) -> Self {
        let blocked = config.plane_size == 1;
        let shape = DeviceShape {
            max_alloc: client.properties().memory.max_page_size as usize,
            cube_dim: config.cube_dim,
            cube_count: if blocked { 1 } else { config.cube_count },
        };

        Self::sized(shape, line_bytes, spec, blocked)
    }

    /// The geometry itself, with the device reduced to its allocation limit and
    /// launch shape so the sizing can be exercised without one.
    ///
    /// The pool is always the whole buffer, so every window has somewhere cold
    /// to come back to. A window pinned to its own bytes revisits them one line
    /// further along each pass, which is a stationary probe: on both a 32 MiB
    /// and an 18 MiB last level cache that reads 10% and writes 20% high.
    ///
    /// The launch shrinks with the window instead of the window growing to fill
    /// the launch. A small window measured with the full dispatch would either
    /// hand many threads the same line or be padded back up to a large one, and
    /// neither is the small-kernel behaviour the curve exists to describe: a
    /// kernel that moves little has little in flight, and that is precisely
    /// what limits it.
    fn sized(shape: DeviceShape, line_bytes: usize, spec: MemorySpec, blocked: bool) -> Self {
        let buffers = spec.access.buffers() as usize;
        let window_bytes = (spec.bytes.min(usize::MAX as u64) as usize) / buffers;

        let cold_lines = (shape.max_alloc.min(DEFAULT_BUFFER_BYTES as usize) / line_bytes).max(1);
        let window_lines = (window_bytes / line_bytes).max(1).min(cold_lines);

        let pool_lines = cold_lines;

        let cube_count = (window_lines / shape.cube_dim).clamp(1, shape.cube_count);

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
    use cubecl_runtime::throughput::MemoryAccess;

    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;

    /// Half a gigabyte of allocation, 256-thread cubes, and a full dispatch of
    /// 2048 cubes.
    const DEVICE: DeviceShape = DeviceShape {
        max_alloc: 512 * MB,
        cube_dim: 256,
        cube_count: 2048,
    };

    /// A probe of that device, in the 16-byte lines a `vec4` of f32 comes in.
    fn probe(working_set: usize, access: MemoryAccess) -> MemoryProbe {
        let spec = MemorySpec::new(access, working_set as u64);
        MemoryProbe::sized(DEVICE, 16, spec, false)
    }

    #[test]
    fn the_pool_stays_large_however_small_the_window() {
        // 256 KiB of traffic, but still half a gigabyte to be cold against.
        let small = probe(256 * KB, MemoryAccess::Read);
        assert_eq!(small.pool_lines, 512 * MB / 16);
        assert_eq!(small.window_lines, 256 * KB / 16);

        // A copy splits its working set across two buffers, so the same traffic
        // is half the window per buffer.
        let copy = probe(256 * KB, MemoryAccess::Copy);
        assert_eq!(copy.pool_lines, 512 * MB / 16);
        assert_eq!(copy.window_lines, 128 * KB / 16);
    }

    /// A window pinned to its own bytes would revisit them one line further
    /// along each pass, which is a stationary probe reading its own cache.
    /// Nothing below the top of the sweep may stop walking.
    #[test]
    fn no_window_short_of_the_pool_stops_walking() {
        for bytes in [256 * KB, 64 * MB, 128 * MB, 256 * MB] {
            let probe = probe(bytes, MemoryAccess::Read);
            assert_eq!(probe.pool_lines, 512 * MB / 16, "at {bytes} bytes");
            assert!(probe.window_lines < probe.pool_lines, "at {bytes} bytes");
            assert_eq!(probe.buffer_bytes, 512 * MB, "at {bytes} bytes");
        }
    }

    #[test]
    fn a_small_window_carries_the_passes_that_walk_the_pool() {
        // 8 KiB of a half-gigabyte pool: 65536 passes to come back round.
        let small = probe(8 * KB, MemoryAccess::Read);
        assert_eq!(small.min_iterations(), 512 * MB / (8 * KB));

        // Only the top of the sweep, where the window is the pool, needs one.
        let whole = probe(512 * MB, MemoryAccess::Read);
        assert_eq!(whole.min_iterations(), 1);
    }

    #[test]
    fn walking_the_pool_costs_the_pool_whatever_the_window() {
        // Every pass moves one window, so the traffic a launch must carry is
        // the pool, and a small window buys passes rather than time.
        for bytes in [8 * KB, 256 * KB, 4 * MB] {
            let probe = probe(bytes, MemoryAccess::Read);
            let walked = probe.min_iterations() * probe.window_lines;
            assert_eq!(walked, probe.pool_lines, "at {bytes} bytes");
        }
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
        let shape = DeviceShape {
            max_alloc: 4 * MB,
            ..DEVICE
        };
        let spec = MemorySpec::new(MemoryAccess::Read, 512 * MB as u64);
        let probe = MemoryProbe::sized(shape, 16, spec, false);

        assert_eq!(probe.buffer_bytes, 4 * MB);
        assert_eq!(probe.window_lines, probe.pool_lines);
    }
}

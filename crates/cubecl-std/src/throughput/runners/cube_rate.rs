use cubecl::prelude::*;
use cubecl_core as cubecl;

/// The fewest cubes one launch of the probe spawns: enough that its launch overhead is a
/// rounding error against them.
const MIN_CUBES: u32 = 128 * 128;

/// Cubes along each side of a launch's grid, which every runtime spawns.
const GRID_SIDE: u32 = 4096;

/// What a read past the last-level cache covers where the runtime reports no cache: twice the
/// largest a GPU carries, the 128 MB of an RTX PRO 6000's L2.
const COLD_BYTES: usize = 256 << 20;

/// The probe of [`ThroughputMode::CubeRoundTrips`](cubecl_runtime::throughput::ThroughputMode::CubeRoundTrips):
/// launches of a grid of cubes of `units` units, each unit reading one value from global memory
/// and writing it back, counted in cubes.
///
/// A round trip and not an empty cube, because a cube that touches no memory retires at once:
/// what bounds a grid of small cubes is how many of them the device keeps waiting on memory at
/// a time, which only a cube that waits shows.
pub fn build_kernel(
    client: &cubecl_runtime::client::Client,
    units: u32,
) -> crate::throughput::KernelSampler {
    let client = client.clone();
    let hardware = &client.properties().hardware;
    let units = units.clamp(1, hardware.max_units_per_cube);
    // Every round trip goes past the last-level cache: a cube waiting on a hit retires long
    // before one waiting on memory, and a grid of small cubes is bound by the memory's wait.
    let cold = hardware
        .last_level_cache_size
        .map_or(COLD_BYTES, |cache| 2 * cache);
    let per_cube = units as usize * core::mem::size_of::<u32>();
    let cubes = cold.div_ceil(per_cube).max(MIN_CUBES as usize) as u32;
    let grid = (cubes.min(GRID_SIDE), cubes.div_ceil(GRID_SIDE));
    let cubes = grid.0 * grid.1;
    let values = (cubes * units) as usize;
    let input = client.empty(values * core::mem::size_of::<u32>());
    let output = client.empty(values * core::mem::size_of::<u32>());
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();

        let profiled = client.profile(
            || unsafe {
                for _ in 0..iterations {
                    round_trip_cube::launch_unchecked(
                        &client,
                        cubecl_core::CubeCount::Static(grid.0, grid.1, 1),
                        cubecl_core::server::CubeDim::new_1d(units),
                        cubecl_core::frontend::BufferArg::from_raw_parts(input.clone(), values),
                        cubecl_core::frontend::BufferArg::from_raw_parts(output.clone(), values),
                    );
                }
            },
            "cube_rate",
        );

        // As the launch probe: the device's window where it timed one, else the host clock
        // over the same launches.
        let measured = match profiled {
            Ok((_, duration)) => {
                cubecl_core::future::block_on(duration.into_future()).map(|ticks| ticks.duration())
            }
            Err(_) => None,
        };
        measured.unwrap_or_else(|| {
            let _ = cubecl_core::future::block_on(client.sync());
            start.elapsed()
        })
    });

    crate::throughput::KernelSampler::new(sample, cubes as usize, 1)
}

/// Every unit reads its own value and writes it back: one round trip to global memory a unit,
/// the least a cube that does any work makes.
#[cube(launch_unchecked)]
fn round_trip_cube(input: &[u32], output: &mut [u32]) {
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

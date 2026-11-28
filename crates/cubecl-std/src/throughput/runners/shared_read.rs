use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Units of a cube where the device launches that many, and cubes of a launch: enough units to
/// fill any device many times over.
const UNITS: u32 = 256;
const CUBES: u32 = 8 * 1024;

/// Vectors each unit reads from shared memory for the one it writes to global memory.
const READS_PER_UNIT: u32 = 256;

/// Floats in a vector the probe reads, the width every runtime loads.
const VECTOR: usize = 4;

/// The probe of
/// [`ThroughputMode::SharedRead`](cubecl_runtime::throughput::ThroughputMode::SharedRead): every
/// cube of a grid that fills the device fills a shared array, a vector a unit, and every unit
/// reads its way around it. Counted in floats read from shared memory.
pub fn build_kernel(client: &cubecl_runtime::client::Client) -> crate::throughput::KernelSampler {
    let client = client.clone();
    let cube_units = UNITS.min(client.properties().hardware.max_units_per_cube);
    let units = (cube_units * CUBES) as usize;
    let float = core::mem::size_of::<f32>();
    let output = client.empty(units * VECTOR * float);
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        let profiled = client.profile(
            || unsafe {
                for _ in 0..iterations {
                    shared_read::launch_unchecked(
                        &client,
                        cubecl_core::CubeCount::Static(CUBES, 1, 1),
                        cubecl_core::server::CubeDim::new_1d(cube_units),
                        VECTOR,
                        cubecl_core::frontend::BufferArg::from_raw_parts(
                            output.clone(),
                            units * VECTOR,
                        ),
                        cube_units as usize,
                        cubecl_core::ir::ElemType::Float(cubecl_core::ir::FloatKind::F32),
                    );
                }
            },
            "shared_read",
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

    let floats_read = units * READS_PER_UNIT as usize * VECTOR;
    crate::throughput::KernelSampler::new(sample, floats_read, 1)
}

/// Every unit writes its own vector of a shared array, then, once the cube has synced, sums
/// [`READS_PER_UNIT`] vectors of it starting at its own and stepping one along, and writes the
/// sum, so consecutive units read consecutive vectors on every step.
///
/// - `output`: one vector a unit, its sum.
/// - `shared_vectors`: the vectors of the shared array, one a unit of the cube.
#[cube(launch_unchecked)]
fn shared_read<F: Float, N: Size>(
    output: &mut [Vector<F, N>],
    #[comptime] shared_vectors: usize,
    #[define(F)] _dtype: ElemType,
) {
    let mut shared = Shared::<[Vector<F, N>]>::new_slice(shared_vectors);
    shared[UNIT_POS as usize] = Vector::<F, N>::cast_from(UNIT_POS);
    sync_cube();
    let mut sum = Vector::<F, N>::cast_from(0u32);
    for i in 0..READS_PER_UNIT as usize {
        sum += shared[(UNIT_POS as usize + i) % shared_vectors];
    }
    output[ABSOLUTE_POS] = sum;
}

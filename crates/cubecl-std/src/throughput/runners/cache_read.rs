use cubecl::prelude::*;
use cubecl_core as cubecl;

/// The smallest window a probe reads: one the last-level cache of every GPU holds whole.
pub const MIN_WINDOW_BYTES: u64 = 1 << 20;

/// The largest window a probe reads: past every GPU's last-level cache, where the read is
/// memory's.
pub const MAX_WINDOW_BYTES: u64 = 256 << 20;

/// Vectors each unit reads for the one it writes, so the writes are noise against the reads.
const READS_PER_UNIT: u32 = 64;

/// Units of a cube, and cubes of a launch: enough units to fill any device many times over.
const UNITS: u32 = 256;
const CUBES: u32 = 16 * 1024;

/// Floats in a vector the probe reads, the width every runtime loads.
const VECTOR: usize = 4;

/// The probe of
/// [`ThroughputMode::CacheRead`](cubecl_runtime::throughput::ThroughputMode::CacheRead): every
/// unit of a grid that fills the device reads its share of one window of `window_bytes`, over
/// and over, as the cubes of a kernel read again what other cubes read. Counted in floats read.
pub fn build_kernel(
    client: &cubecl_runtime::client::Client,
    window_bytes: u64,
) -> crate::throughput::KernelSampler {
    let client = client.clone();
    let float = core::mem::size_of::<f32>();
    let window_bytes = window_bytes.clamp(MIN_WINDOW_BYTES, MAX_WINDOW_BYTES) as usize;
    let window_vectors = window_bytes / (float * VECTOR);
    let units = (UNITS * CUBES) as usize;
    let input = client.empty(window_bytes);
    let output = client.empty(units * VECTOR * float);
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let start = cubecl_common::profile::Instant::now();
        let profiled = client.profile(
            || unsafe {
                for _ in 0..iterations {
                    cache_read::launch_unchecked(
                        &client,
                        cubecl_core::CubeCount::Static(CUBES, 1, 1),
                        cubecl_core::server::CubeDim::new_1d(UNITS),
                        VECTOR,
                        cubecl_core::frontend::BufferArg::from_raw_parts(
                            input.clone(),
                            window_vectors,
                        ),
                        cubecl_core::frontend::BufferArg::from_raw_parts(output.clone(), units),
                        cubecl_core::ir::ElemType::Float(cubecl_core::ir::FloatKind::F32),
                    );
                }
            },
            "cache_read",
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

/// Every unit sums [`READS_PER_UNIT`] vectors of the window and writes the sum. Unit `u` reads
/// vectors `u + i · stride`, wrapping around the window, the stride a `READS_PER_UNIT`th of it:
/// consecutive units read consecutive vectors on every step, and a unit's own reads lie a whole
/// stretch of the window apart, so a vector is read again only after the rest of the window was,
/// and the window's size, not a unit's reach, decides which cache serves it.
#[cube(launch_unchecked)]
fn cache_read<F: Float, N: Size>(
    window: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    #[define(F)] _dtype: ElemType,
) {
    let stride = window.len() / READS_PER_UNIT as usize + 1;
    let mut sum = Vector::<F, N>::cast_from(0u32);
    #[unroll]
    for i in 0..READS_PER_UNIT as usize {
        sum += window[(ABSOLUTE_POS + i * stride) % window.len()];
    }
    output[ABSOLUTE_POS] = sum;
}

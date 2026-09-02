use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::CubeDim,
    throughput::{
        ComputeCmmaConfig, DEFAULT_BUFFER_BYTES, KernelConfig, MemoryAccess, MemoryCurve,
        MemoryPoint, MemorySpec, ThroughputBenchmarker, ThroughputKey, ThroughputMode,
        ThroughputValue, sweep_size, working_set_sweep,
    },
    tune::{Bounds, Thresholds, Work, calculate_bounds},
};

use crate::throughput::{
    compute_cmma, compute_direct, launch_overhead, memory_direct, memory_read, memory_write,
};

/// Independent cube positions each CPU worker interleaves, so a compute
/// pass pipelines past instruction latency instead of serializing on one
/// dependency chain. A depth, not a machine guess: a handful hides any
/// core's fma latency, excess is free because the iteration budget is
/// time-calibrated, and nothing about the launch scales with it. Memory
/// probes with blocked addressing pin their own count back to one (see
/// [`MemoryProbe::new`](crate::throughput::memory_probe::MemoryProbe::new)).
const CPU_CHAIN_DEPTH: usize = 64;

/// Measure peak throughput on `device` for each of the given `keys`.
pub fn device_throughput<R: Runtime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<ThroughputValue> {
    let client = R::client(device);
    keys.iter()
        .map(|key| measure_peak_throughput::<R>(&client, *key))
        .collect()
}

/// Measure the memory ceiling across a range of working sets, from a few
/// kilobytes up to as much as the device will allocate.
///
/// One point per size in [`working_set_sweep`], each measured and cached
/// exactly like the single-size probe — [`measure_peak_throughput`] with a
/// [`ThroughputMode::Memory`] key — so a curve costs one probe per size on the
/// first run and nothing afterwards.
///
/// Native only, panics on WASM
pub fn measure_memory_curve<R: Runtime>(
    client: &ComputeClient<R>,
    access: MemoryAccess,
) -> MemoryCurve {
    let points = sweep::<R>(client, access, |bytes| {
        ThroughputMode::Memory(MemorySpec::new(access, bytes))
    });

    MemoryCurve::new(access, points)
}

/// One measured point per size in [`working_set_sweep`], each cached exactly
/// like the single-size probe, so a curve costs one probe per size on the
/// first run and nothing afterwards.
fn sweep<R: Runtime>(
    client: &ComputeClient<R>,
    access: MemoryAccess,
    mode: impl Fn(u64) -> ThroughputMode,
) -> alloc::vec::Vec<MemoryPoint> {
    working_set_sweep(working_set_cap(client, access))
        .into_iter()
        .map(|bytes| MemoryPoint {
            bytes,
            value: measure_peak_throughput::<R>(client, ThroughputKey { mode: mode(bytes) }),
        })
        .collect()
}

/// The largest working set `access` can be probed at: as much as one buffer can
/// hold, times the buffers the access touches.
fn working_set_cap<R: Runtime>(client: &ComputeClient<R>, access: MemoryAccess) -> u64 {
    let max_alloc = client.properties().memory.max_page_size;

    DEFAULT_BUFFER_BYTES.min(max_alloc) * access.buffers()
}

/// Computes the peak throughput for a given runtime and key.
///
/// Native only, panics on WASM
pub fn measure_peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> ThroughputValue {
    // A throughput probe is a measurement: inside a dry run its launches must
    // still execute, or they would be timed anyway and cache a garbage peak in
    // the device-level throughput store. The guard is read where the launch is
    // issued, which for these is this thread.
    let _measurement = cubecl_runtime::dry_run::RealRun::new();

    let launch_config = launch_config(client, key.dtype());

    let candidates = match key.mode {
        ThroughputMode::ComputeDirect { .. } => arithmetic_widths(client, key.dtype())
            .into_iter()
            .map(|vector_size| {
                let config = LaunchConfig {
                    vector_size,
                    ..launch_config
                };
                compute_direct::build_kernel(client, key, config)
            })
            .collect(),
        ThroughputMode::ComputeCmma {
            dtype,
            config: cmma_config,
        } => {
            if !implements_cmma(client, dtype, cmma_config) {
                return ThroughputValue::ZERO;
            }
            alloc::vec![compute_cmma::build_kernel(
                client,
                key,
                cmma_config,
                launch_config
            )]
        }
        ThroughputMode::Memory(spec) => alloc::vec![match spec.access {
            MemoryAccess::Copy => memory_direct::build_kernel(client, key, launch_config, spec),
            MemoryAccess::Read => memory_read::build_kernel(client, key, launch_config, spec),
            MemoryAccess::Write => memory_write::build_kernel(client, key, launch_config, spec),
        }],
        ThroughputMode::Launch => {
            alloc::vec![launch_overhead::build_kernel(client, key, launch_config)]
        }
    };

    let value = client.measure_throughput(key, || fastest_shape(candidates));

    client.memory_cleanup();

    value
}

/// Calculates roofline autotune bounds for a given [`Work`] amount and compute throughput key.
///
/// Measures compute and memory peak throughputs along with launch overhead for the runtime client.
pub fn roofline_bounds<R: Runtime>(
    client: &ComputeClient<R>,
    compute_key: ThroughputKey,
    work: Work,
    thresholds: Thresholds,
) -> Bounds {
    // Past what the device will allocate the probe measures the cap regardless,
    // so capping the ask keeps one cache entry rather than one per kernel.
    let access = MemoryAccess::Copy;
    let footprint = (work.bytes as u64).min(working_set_cap(client, access));
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory(MemorySpec::new(access, sweep_size(footprint))),
    };
    let launch_key = ThroughputKey {
        mode: ThroughputMode::Launch,
    };

    Bounds {
        bounds: calculate_bounds(
            work,
            thresholds,
            &measure_peak_throughput(client, compute_key),
            &measure_peak_throughput(client, memory_key),
            &memory_key,
        ),
        launch_overhead: measure_peak_throughput(client, launch_key).duration_per_op(),
    }
}

/// What the device answers fastest, of the shapes a probe can be launched in.
///
/// A rate that is not finite is a shape that did not run rather than a slow
/// one, so it does not stand as an answer; where none of them ran there is no
/// peak to report.
fn fastest_shape(candidates: alloc::vec::Vec<KernelConfig>) -> ThroughputValue {
    candidates
        .into_iter()
        .map(ThroughputBenchmarker::sample)
        .filter(|value| value.ops_per_s().is_finite())
        .max_by(|a, b| a.ops_per_s().total_cmp(&b.ops_per_s()))
        .unwrap_or(ThroughputValue::ZERO)
}

/// The vector widths an arithmetic probe is measured at.
///
/// [`io_optimized_vector_sizes`](ComputeClient::io_optimized_vector_sizes)
/// orders widest first because it describes load and store instructions, and
/// an arithmetic probe issues none. Which width retires the most is a property
/// of the device — a SIMD CPU needs the widest, while on a scalar-lane GPU the
/// lane layout a wide vector imposes costs more shuffling than its packing
/// saves — so the probe measures every width the device offers.
fn arithmetic_widths<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
) -> alloc::vec::Vec<usize> {
    let widths: alloc::vec::Vec<usize> = client.io_optimized_vector_sizes(dtype.size()).collect();

    if widths.is_empty() {
        alloc::vec![1]
    } else {
        widths
    }
}

/// Whether the device implements the cooperative matrix the probe launches.
///
/// A non-empty capability list says the device has tensor hardware, not that it
/// has this shape on it, and launching a shape it lacks measures nothing while
/// looking like any other number. The manual `mma` list is deliberately not
/// consulted: the probe issues `cmma::execute`.
fn implements_cmma<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    config: ComputeCmmaConfig,
) -> bool {
    client.properties().features.matmul.cmma.iter().any(|it| {
        it.a_type == dtype
            && it.b_type == dtype
            && it.cd_type == config.accumulator_type
            && it.m as usize == config.cmma_dims.m
            && it.n as usize == config.cmma_dims.n
            && it.k as usize == config.cmma_dims.k
    })
}

/// Hardware execution parameters for launching a compute kernel.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    /// The number of threads per cube.
    pub cube_dim: usize,
    /// The total number of cubes to dispatch.
    pub cube_count: usize,
    /// The vectorization factor (e.g., 4 for `vec4` operations).
    pub vector_size: usize,
    /// The number of threads in a hardware execution plane.
    pub plane_size: usize,
}

fn launch_config<R: Runtime>(client: &ComputeClient<R>, dtype: ElemType) -> LaunchConfig {
    let hardware = &client.properties().hardware;

    let plane_size = hardware.plane_size_max.max(1);
    let vector_size = client
        .io_optimized_vector_sizes(dtype.size())
        .next()
        .unwrap_or(1);

    // A CPU has no SMs, so `sms * 32` cubes is the wrong grid to size from:
    // a cube's units are its real dispatched workers here, while its cube
    // count is only a loop inside each of them. `num_cpu_cores` units, one
    // per core, is the real worker count.
    if let Some(cores) = hardware.num_cpu_cores {
        return LaunchConfig {
            cube_dim: cores as usize,
            cube_count: CPU_CHAIN_DEPTH,
            vector_size,
            plane_size: plane_size as usize,
        };
    }

    let requested = (hardware.max_units_per_cube / plane_size * plane_size)
        .max(plane_size)
        .min(hardware.max_cube_dim.0);

    let cube_dim = CubeDim::new(client, requested as usize).num_elems();

    let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
    let cube_count = (sms * 32).min(hardware.max_cube_count.0);

    LaunchConfig {
        cube_dim: cube_dim as usize,
        cube_count: cube_count as usize,
        vector_size,
        plane_size: plane_size as usize,
    }
}

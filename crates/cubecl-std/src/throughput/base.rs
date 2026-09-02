use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::CubeDim,
    throughput::{
        ComputeCmmaConfig, DEFAULT_BUFFER_BYTES, KernelConfig, MemoryAccess, MemoryCurve,
        MemoryPoint, MemorySpec, ThroughputBenchmarker, ThroughputError, ThroughputKey,
        ThroughputMode, ThroughputValue, sweep_size, working_set_sweep,
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

/// Units a GPU probe asks for. A wider cube measures no faster, and makes the
/// memory probes report several times the bus rate.
const PROBE_UNITS_PER_CUBE: u32 = 256;

/// Measure peak throughput on `device` for each of the given `keys`.
pub fn device_throughput<R: Runtime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<Result<ThroughputValue, ThroughputError>> {
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
        .filter_map(|bytes| {
            let key = ThroughputKey { mode: mode(bytes) };

            Some(MemoryPoint {
                bytes,
                value: measure_peak_throughput::<R>(client, key).ok()?,
            })
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
///
/// # Errors
///
/// [`Unsupported`](ThroughputError::Unsupported) where the device implements
/// no such operation, [`NoTiming`](ThroughputError::NoTiming) where it does
/// and reported no elapsed time.
pub fn measure_peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
) -> Result<ThroughputValue, ThroughputError> {
    // A throughput probe is a measurement: inside a dry run its launches must
    // still execute, or they would be timed anyway and cache a garbage peak in
    // the device-level throughput store. The guard is read where the launch is
    // issued, which for these is this thread.
    let _measurement = cubecl_runtime::dry_run::RealRun::new();

    let launch_config = launch_config(client, key.dtype());

    let candidates = match key.mode {
        ThroughputMode::ComputeDirect { dtype } => {
            // A type the backend cannot lower panics rather than answering.
            if !client.properties().features.supports_type(dtype) {
                return Err(ThroughputError::Unsupported);
            }

            arithmetic_widths(client, dtype)
                .into_iter()
                .map(|vector_size| {
                    let config = LaunchConfig {
                        vector_size,
                        ..launch_config
                    };
                    compute_direct::build_kernel(client, key, config)
                })
                .collect()
        }
        ThroughputMode::ComputeCmma {
            dtype,
            config: cmma_config,
        } => {
            if !implements_cmma(client, dtype, cmma_config) {
                return Err(ThroughputError::Unsupported);
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

    // No ceiling to bound against, which `time_at_peak` already declines.
    let no_ceiling = ThroughputValue::ZERO;

    Bounds {
        bounds: calculate_bounds(
            work,
            thresholds,
            &measure_peak_throughput(client, compute_key).unwrap_or(no_ceiling),
            &measure_peak_throughput(client, memory_key).unwrap_or(no_ceiling),
            &memory_key,
        ),
        launch_overhead: measure_peak_throughput(client, launch_key)
            .map(|value| value.duration_per_op())
            .unwrap_or_default(),
    }
}

/// What the device answers fastest, of the shapes a probe can be launched in.
///
/// A rate that is not finite is a shape that did not run, not a slow one.
fn fastest_shape(
    candidates: alloc::vec::Vec<KernelConfig>,
) -> Result<ThroughputValue, ThroughputError> {
    candidates
        .into_iter()
        .map(ThroughputBenchmarker::sample)
        .filter(|value| value.ops_per_s().is_finite())
        .max_by(|a, b| a.ops_per_s().total_cmp(&b.ops_per_s()))
        .ok_or(ThroughputError::NoTiming)
}

/// The vector widths an arithmetic probe is measured at.
///
/// `io_optimized_vector_sizes` is ordered for the loads and stores this probe
/// issues none of, and its widest is not the fastest on every device.
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
/// A non-empty capability list says the device has tensor hardware, not this
/// shape of it. `mma` is not consulted: the probe issues `cmma::execute`.
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
    /// The cube the probe launches, resolved once so `ops_count` cannot
    /// describe a launch that did not happen.
    pub cube_dim: CubeDim,
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
    let (units, cube_count) = match hardware.num_cpu_cores {
        Some(cores) => (cores, CPU_CHAIN_DEPTH as u32),
        None => {
            let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
            (
                PROBE_UNITS_PER_CUBE,
                (sms * 32).min(hardware.max_cube_count.0),
            )
        }
    };

    LaunchConfig {
        cube_dim: CubeDim::new(client, units as usize),
        cube_count: cube_count as usize,
        vector_size,
        plane_size: plane_size as usize,
    }
}

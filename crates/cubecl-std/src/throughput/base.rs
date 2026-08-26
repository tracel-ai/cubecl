use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::CubeDim,
    throughput::{
        DEFAULT_BUFFER_BYTES, MemoryAccess, MemoryCurve, MemoryPoint, ThroughputKey,
        ThroughputMode, ThroughputValue, working_set_sweep,
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
/// [`ThroughputMode::MemoryWorkingSet`] key — so a curve costs one probe per
/// size on the first run and nothing afterwards.
///
/// Native only, panics on WASM
pub fn measure_memory_curve<R: Runtime>(
    client: &ComputeClient<R>,
    access: MemoryAccess,
) -> MemoryCurve {
    let points = sweep::<R>(client, access, |bytes| ThroughputMode::MemoryWorkingSet {
        access,
        bytes,
    });

    MemoryCurve::new(access, points)
}

/// Measure the same range of working sets where each one already sits, rather
/// than walking it until every pass finds it evicted.
///
/// The ceiling for a kernel that comes back to a tile it has already touched,
/// which against [`measure_memory_curve`] can report several hundred percent
/// and be right to. Past the cache the two answer the same question and the
/// curves should meet.
///
/// Native only, panics on WASM
pub fn measure_resident_curve<R: Runtime>(
    client: &ComputeClient<R>,
    access: MemoryAccess,
) -> MemoryCurve {
    let points = sweep::<R>(client, access, |bytes| ThroughputMode::MemoryResident {
        access,
        bytes,
    });

    MemoryCurve::resident(access, points)
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

    let kernel_config = match key.mode {
        ThroughputMode::ComputeDirect { .. } => {
            compute_direct::build_kernel(client, key, launch_config)
        }
        ThroughputMode::ComputeCmma {
            config: cmma_config,
            ..
        } => {
            if client.properties().features.matmul.cmma.is_empty() {
                return ThroughputValue::ZERO;
            }
            compute_cmma::build_kernel(client, key, cmma_config, launch_config)
        }
        ThroughputMode::Memory
        | ThroughputMode::MemoryRead
        | ThroughputMode::MemoryWrite
        | ThroughputMode::MemoryWorkingSet { .. }
        | ThroughputMode::MemoryResident { .. } => {
            // The memory modes differ only in what they ask of the probe, and
            // `memory_probe` is the one place that mapping lives.
            let spec = key
                .mode
                .memory_probe()
                .expect("A memory mode describes a probe");

            match spec.access {
                MemoryAccess::Copy => memory_direct::build_kernel(client, key, launch_config, spec),
                MemoryAccess::Read => memory_read::build_kernel(client, key, launch_config, spec),
                MemoryAccess::Write => memory_write::build_kernel(client, key, launch_config, spec),
            }
        }
        ThroughputMode::Launch => launch_overhead::build_kernel(client, key, launch_config),
    };

    let value = client.measure_throughput(key, kernel_config);

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
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
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

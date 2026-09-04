use alloc::string::String;
use cubecl_core::ir::ElemType;
use cubecl_environment::{collections::HashMap, sync::Mutex};
use cubecl_runtime::{
    client::Client,
    runtime::Runtime,
    server::CubeDim,
    throughput::{
        ComputeCmmaConfig, KernelConfig, MemoryAccess, MemoryCurve, MemoryPoint, MemorySpec,
        ThroughputBenchmarker, ThroughputError, ThroughputKey, ThroughputMode, ThroughputValue,
        sweep_size, working_set_sweep,
    },
    tune::{Bounds, Thresholds, Work, calculate_bounds},
};

use crate::throughput::{
    compute_cmma, compute_direct, launch_overhead, memory_direct, memory_probe, memory_read,
    memory_write,
};

/// Independent cube positions each CPU worker interleaves, so a compute
/// pass pipelines past instruction latency instead of serializing on one
/// dependency chain. A depth, not a machine guess: a handful hides any
/// core's fma latency, excess is free because the iteration budget is
/// time-calibrated, and nothing about the launch scales with it. Memory
/// probes with blocked addressing pin their own count back to one (see
/// [`MemoryProbe::new`](crate::throughput::memory_probe::MemoryProbe::new)).
const CPU_CHAIN_DEPTH: usize = 64;

/// Worker counts a memory probe is swept over before the fastest is kept, once
/// per device and access. Five halvings reach a sixteenth of the cores, and
/// each one costs a warmup and a sample.
const MEMORY_WORKER_SHAPES: usize = 5;

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
        .map(|key| measure_peak_throughput(&client, *key))
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
pub fn measure_memory_curve(client: &Client, access: MemoryAccess) -> MemoryCurve {
    let points = sweep(client, access, |bytes| {
        ThroughputMode::Memory(MemorySpec::new(access, bytes))
    });

    MemoryCurve::new(access, points)
}

/// One measured point per size in [`working_set_sweep`], each cached exactly
/// like the single-size probe, so a curve costs one probe per size on the
/// first run and nothing afterwards.
fn sweep(
    client: &Client,
    access: MemoryAccess,
    mode: impl Fn(u64) -> ThroughputMode,
) -> alloc::vec::Vec<MemoryPoint> {
    working_set_sweep(working_set_cap(client, access))
        .into_iter()
        .filter_map(|bytes| {
            let key = ThroughputKey { mode: mode(bytes) };

            Some(MemoryPoint {
                bytes,
                value: measure_peak_throughput(client, key).ok()?,
            })
        })
        .collect()
}

/// The largest working set `access` can be probed at: the largest window one
/// buffer holds, times the buffers the access touches.
fn working_set_cap(client: &Client, access: MemoryAccess) -> u64 {
    let max_alloc = client.properties().memory.max_page_size;

    memory_probe::window_cap(max_alloc) * access.buffers()
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
pub fn measure_peak_throughput(
    client: &Client,
    key: ThroughputKey,
) -> Result<ThroughputValue, ThroughputError> {
    // A throughput probe is a measurement: inside a dry run its launches must
    // still execute, or they would be timed anyway and cache a garbage peak in
    // the device-level throughput store. The guard is read where the launch is
    // issued, which for these is this thread.
    let _measurement = cubecl_runtime::dry_run::RealRun::new();

    let value = client.measure_throughput(key, || probe(client, key));

    client.memory_cleanup();

    value
}

/// Measures `key`, in the fastest shape its probe can be launched in.
///
/// # Errors
///
/// [`Unsupported`](ThroughputError::Unsupported) where the device implements
/// no such operation, [`NoTiming`](ThroughputError::NoTiming) where it does
/// and reported no elapsed time.
fn probe(client: &Client, key: ThroughputKey) -> Result<ThroughputValue, ThroughputError> {
    let launch_config = launch_config(client, key.dtype());

    match key.mode {
        ThroughputMode::ComputeDirect { dtype } => {
            // A type the backend cannot lower panics rather than answering.
            if !client.properties().features.supports_type(dtype) {
                return Err(ThroughputError::Unsupported);
            }

            let shapes = arithmetic_widths(client, dtype)
                .into_iter()
                .map(|vector_size| LaunchConfig {
                    vector_size,
                    ..launch_config
                })
                .collect();

            fastest_shape(shapes, |config| {
                compute_direct::build_kernel(client, key, config)
            })
            .map(|(value, _)| value)
        }
        ThroughputMode::ComputeCmma {
            dtype,
            config: cmma_config,
        } => {
            if !implements_cmma(client, dtype, cmma_config) {
                return Err(ThroughputError::Unsupported);
            }

            fastest_shape(alloc::vec![launch_config], |config| {
                compute_cmma::build_kernel(client, key, cmma_config, config)
            })
            .map(|(value, _)| value)
        }
        ThroughputMode::Memory(spec) => {
            let (value, fastest) = fastest_shape(
                memory_shapes(client, launch_config, spec.access),
                |config| match spec.access {
                    MemoryAccess::Copy => memory_direct::build_kernel(client, key, config, spec),
                    MemoryAccess::Read => memory_read::build_kernel(client, key, config, spec),
                    MemoryAccess::Write => memory_write::build_kernel(client, key, config, spec),
                },
            )?;

            remember_workers(client, spec.access, fastest.cube_dim.num_elems());

            Ok(value)
        }
        ThroughputMode::Launch => fastest_shape(alloc::vec![launch_config], |config| {
            launch_overhead::build_kernel(client, key, config)
        })
        .map(|(value, _)| value),
    }
}

/// Calculates roofline autotune bounds for a given [`Work`] amount and compute throughput key.
///
/// Measures compute and memory peak throughputs along with launch overhead for the runtime client.
pub fn roofline_bounds(
    client: &Client,
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

/// What the device answers fastest, of the shapes a probe can be launched in,
/// and the shape that answered it.
///
/// Built one at a time and dropped on the way out: a memory probe's pool is a
/// large fraction of what the device will allocate, and holding every shape's
/// at once would ask for several of them.
///
/// A rate that is not finite is a shape that did not run, not a slow one.
fn fastest_shape(
    shapes: alloc::vec::Vec<LaunchConfig>,
    build: impl Fn(LaunchConfig) -> KernelConfig,
) -> Result<(ThroughputValue, LaunchConfig), ThroughputError> {
    shapes
        .into_iter()
        .map(|shape| (ThroughputBenchmarker::sample(build(shape)), shape))
        .filter(|(value, _)| value.ops_per_s().is_finite())
        .max_by(|(a, _), (b, _)| a.ops_per_s().total_cmp(&b.ops_per_s()))
        .ok_or(ThroughputError::NoTiming)
}

/// The launch shapes a memory probe is measured in.
///
/// A GPU keeps the cube it was pinned to: a wider one measures no faster, and
/// makes these probes report several times the bus rate.
///
/// A CPU sweeps its worker count once per access and then keeps the count that
/// won, at every working set. That count does not move with the window: four
/// workers win all seventeen read points of a Ryzen 7 5700X's curve, by 10 to
/// 22%, and the whole ranking barely shifts across them. Sweeping every point
/// instead costs the curve 2.7x for an answer it already has.
fn memory_shapes(
    client: &Client,
    config: LaunchConfig,
    access: MemoryAccess,
) -> alloc::vec::Vec<LaunchConfig> {
    if client.properties().hardware.num_cpu_cores.is_none() {
        return alloc::vec![config];
    }

    if let Some(units) = remembered_workers(client, access) {
        return alloc::vec![with_units(client, config, units)];
    }

    worker_counts(config.cube_dim.num_elems() as usize)
        .into_iter()
        .map(|units| with_units(client, config, units as u32))
        .collect()
}

/// Worker counts a CPU probe is swept over, the full launch first and halvings
/// of it after.
///
/// How many threads saturate the memory system is a property of the controller
/// rather than of the core count: one per hardware thread reads 44 GB/s of a
/// Ryzen 7 5700X's 51.2, where a quarter of them reads 50. Little's law derives
/// the count from the bandwidth, which is the thing being measured, so it is
/// swept instead.
fn worker_counts(units: usize) -> alloc::vec::Vec<usize> {
    core::iter::successors(Some(units.max(1)), |units| (*units > 1).then(|| units / 2))
        .take(MEMORY_WORKER_SHAPES)
        .collect()
}

fn with_units(client: &Client, config: LaunchConfig, units: u32) -> LaunchConfig {
    LaunchConfig {
        cube_dim: CubeDim::new(client, units as usize),
        ..config
    }
}

/// The worker count each device streams each access fastest at, once one probe
/// of that access has swept for it.
static SATURATING_WORKERS: Mutex<Option<HashMap<(String, MemoryAccess), u32>>> = Mutex::new(None);

fn remembered_workers(client: &Client, access: MemoryAccess) -> Option<u32> {
    let workers = SATURATING_WORKERS.lock();
    let workers = workers.as_ref()?;

    workers.get(&(device_key(client), access)).copied()
}

fn remember_workers(client: &Client, access: MemoryAccess, units: u32) {
    let mut workers = SATURATING_WORKERS.lock();

    workers
        .get_or_insert_with(HashMap::new)
        .insert((device_key(client), access), units);
}

/// Names one device, so two of them do not share a worker count.
fn device_key(client: &Client) -> String {
    let identity = &client.properties().identity;

    alloc::format!(
        "{}_{}_{}",
        client.name(),
        identity.fingerprint,
        identity.name
    )
}

/// The vector widths an arithmetic probe is measured at.
///
/// `io_optimized_vector_sizes` is ordered for the loads and stores this probe
/// issues none of, and its widest is not the fastest on every device.
fn arithmetic_widths(client: &Client, dtype: ElemType) -> alloc::vec::Vec<usize> {
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
fn implements_cmma(client: &Client, dtype: ElemType, config: ComputeCmmaConfig) -> bool {
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

fn launch_config(client: &Client, dtype: ElemType) -> LaunchConfig {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The full launch is measured first, so a device whose peak is there
    /// keeps the shape it had before the sweep existed.
    #[test]
    fn the_sweep_starts_at_the_full_launch_and_halves_to_the_budget() {
        assert_eq!(worker_counts(16), alloc::vec![16, 8, 4, 2, 1]);
        assert_eq!(worker_counts(128), alloc::vec![128, 64, 32, 16, 8]);
    }

    /// Every count is a launch, so one core must not be handed an empty sweep
    /// and reported as untimeable.
    #[test]
    fn one_core_is_still_a_shape() {
        assert_eq!(worker_counts(1), alloc::vec![1]);
    }
}

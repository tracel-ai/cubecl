use cubecl_core::ir::ElemType;
use cubecl_runtime::{client::Client, server::CubeDim};

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

impl LaunchConfig {
    /// The launch a probe of `dtype` is issued in on this device.
    pub(super) fn for_device(client: &Client, dtype: ElemType) -> Self {
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

        Self {
            cube_dim: CubeDim::new(client, units as usize),
            cube_count: cube_count as usize,
            vector_size,
            plane_size: plane_size as usize,
        }
    }

    /// The same launch, dispatched across `units` workers.
    pub(super) fn with_units(self, client: &Client, units: u32) -> Self {
        Self {
            cube_dim: CubeDim::new(client, units as usize),
            ..self
        }
    }
}

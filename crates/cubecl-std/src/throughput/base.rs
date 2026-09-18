use cubecl_core::ir::ElemType;
use cubecl_runtime::{
    client::Client,
    runtime::Runtime,
    throughput::{
        MemoryAccess, MemoryCurve, MemoryPoint, MemorySpec, ThroughputError, ThroughputKey,
        ThroughputMode, ThroughputValue, sweep_size, working_set_sweep,
    },
    tune::{AutotuneBound, Bounds, ResourceBound, Thresholds, Work},
};

use crate::throughput::{
    Arithmetic, CooperativeMatrix, LaunchConfig, PooledProbes, ShapeSweep, WorkerSweep,
    compute_cmma, compute_direct, launch_overhead, memory_direct, memory_probe, memory_read,
    memory_write,
};

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

/// Asynchronously measures peak throughput on `device`.
pub async fn device_throughput_async<R: Runtime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<Result<ThroughputValue, ThroughputError>> {
    let client = R::client(device);
    let mut values = alloc::vec::Vec::with_capacity(keys.len());
    for key in keys {
        values.push(measure_peak_throughput_async(&client, *key).await);
    }
    values
}

/// Measure the memory ceiling across a range of working sets, from a few
/// kilobytes up to as much as the device will allocate.
///
/// One point per size in [`working_set_sweep`], each measured and cached
/// exactly like the single-size probe, so a curve costs one probe per size on
/// the first run and nothing afterwards.
///
/// Native only, panics on WASM
pub fn measure_memory_curve(client: &Client, access: MemoryAccess) -> MemoryCurve {
    #[cfg(target_family = "wasm")]
    {
        let _ = (client, access);
        panic!("measuring memory throughput synchronously is unsupported on wasm")
    }

    #[cfg(not(target_family = "wasm"))]
    cubecl_environment::future::block_on(measure_memory_curve_async(client, access))
}

/// Asynchronously measures the memory ceiling across a range of working sets.
pub async fn measure_memory_curve_async(client: &Client, access: MemoryAccess) -> MemoryCurve {
    let points = {
        // Every point of a sweep asks for the same pool, so the sweep holds one.
        let _pooled = PooledProbes::enter(client);

        sweep(client, access, |bytes| {
            ThroughputMode::Memory(MemorySpec::new(access, bytes))
        })
        .await
    };

    PooledProbes::cleanup_unless_held(client);

    MemoryCurve::new(access, points)
}

async fn sweep(
    client: &Client,
    access: MemoryAccess,
    mode: impl Fn(u64) -> ThroughputMode,
) -> alloc::vec::Vec<MemoryPoint> {
    let mut points = alloc::vec::Vec::new();
    for bytes in working_set_sweep(working_set_cap(client, access)) {
        let key = ThroughputKey { mode: mode(bytes) };
        if let Ok(value) = measure_peak_throughput_async(client, key).await {
            points.push(MemoryPoint { bytes, value });
        }
    }
    points
}

/// The largest working set `access` can be probed at: the largest window one
/// buffer holds, times the buffers the access touches.
fn working_set_cap(client: &Client, access: MemoryAccess) -> u64 {
    let max_alloc = client.properties().memory.max_page_size;

    memory_probe::window_cap(max_alloc) * access.buffers()
}

/// Computes the peak throughput for a given runtime and key.
///
/// Blocks on the measurement, which the browser can't do: there this reports
/// [`Unsupported`](ThroughputError::Unsupported), and
/// [`measure_peak_throughput_async`] is the entry point to await.
///
/// # Errors
///
/// [`Unsupported`](ThroughputError::Unsupported) where the device implements
/// no such operation, [`NoTiming`](ThroughputError::NoTiming) where it does
/// and reported no elapsed time, [`Allocation`](ThroughputError::Allocation)
/// where it has no room for the probe's buffers, [`Launch`](ThroughputError::Launch)
/// where a memory probe's kernel did not run. None of them is cached, so a
/// device that was full is measured the next time it is asked.
pub fn measure_peak_throughput(
    client: &Client,
    key: ThroughputKey,
) -> Result<ThroughputValue, ThroughputError> {
    #[cfg(target_family = "wasm")]
    {
        // A roofline bound built from this on the browser has no peak, and
        // so no time limit; the tune runs without one rather than not at all.
        let _ = (client, key);
        Err(ThroughputError::Unsupported)
    }

    #[cfg(not(target_family = "wasm"))]
    cubecl_environment::future::block_on(measure_peak_throughput_async(client, key))
}

/// Asynchronously computes the peak throughput for `key`.
pub async fn measure_peak_throughput_async(
    client: &Client,
    key: ThroughputKey,
) -> Result<ThroughputValue, ThroughputError> {
    // A throughput probe is a measurement: inside a dry run its launches must
    // still execute, or they would be timed anyway and cache a garbage peak in
    // the device-level throughput store. The guard is read where the launch is
    // issued, which for these is this thread.
    let _measurement = cubecl_runtime::dry_run::RealRun::new();

    let value = client.measure_throughput(key, || probe(client, key)).await;

    PooledProbes::cleanup_unless_held(client);

    value
}

/// Measures `key`, in the fastest shape its probe can be launched in.
async fn probe(client: &Client, key: ThroughputKey) -> Result<ThroughputValue, ThroughputError> {
    let launch_config = LaunchConfig::for_device(client, key.dtype());

    match key.mode {
        ThroughputMode::ComputeDirect { dtype } => {
            // A type the backend cannot lower panics rather than answering.
            if !client.properties().features.supports_type(dtype) {
                return Err(ThroughputError::Unsupported);
            }

            ShapeSweep::new(compute_direct_shapes(client, dtype, launch_config))
                .fastest(|(dtype, config)| async move {
                    Ok(compute_direct::build_kernel(client, dtype, config).await)
                })
                .await
                .map(|(value, _)| value)
        }
        ThroughputMode::ComputeCmma {
            dtype,
            config: cmma_config,
        } => {
            if !CooperativeMatrix::implemented(client, dtype, cmma_config) {
                return Err(ThroughputError::Unsupported);
            }

            ShapeSweep::new(alloc::vec![launch_config])
                .fastest(|config| async move {
                    Ok(compute_cmma::build_kernel(client, key, cmma_config, config).await)
                })
                .await
                .map(|(value, _)| value)
        }
        ThroughputMode::Memory(spec) => {
            let (value, fastest) =
                ShapeSweep::new(WorkerSweep::shapes(client, launch_config, spec.access))
                    .fastest(|config| async move {
                        match spec.access {
                            MemoryAccess::Copy => {
                                memory_direct::build_kernel(client, key, config, spec).await
                            }
                            MemoryAccess::Read => {
                                memory_read::build_kernel(client, key, config, spec).await
                            }
                            MemoryAccess::Write => {
                                memory_write::build_kernel(client, key, config, spec).await
                            }
                        }
                    })
                    .await?;

            WorkerSweep::remember(client, spec.access, fastest.cube_dim.num_elems());

            Ok(value)
        }
        ThroughputMode::Launch => ShapeSweep::new(alloc::vec![launch_config])
            .fastest(|config| async move {
                Ok(launch_overhead::build_kernel(client, key, config).await)
            })
            .await
            .map(|(value, _)| value),
    }
}

/// Every operand type and vector width the arithmetic ceiling for `dtype` is
/// measured over.
fn compute_direct_shapes(
    client: &Client,
    dtype: ElemType,
    launch_config: LaunchConfig,
) -> alloc::vec::Vec<(ElemType, LaunchConfig)> {
    Arithmetic::dtypes(client, dtype)
        .into_iter()
        .flat_map(|dtype| {
            Arithmetic::widths(client, dtype)
                .into_iter()
                .map(move |vector_size| {
                    (
                        dtype,
                        LaunchConfig {
                            vector_size,
                            ..launch_config
                        },
                    )
                })
        })
        .collect()
}

/// Both halves of the roofline for a [`Work`] amount, with the memory ceiling a copy's.
///
/// A copy's traffic runs in both directions, which is the ceiling for a kernel that reads
/// and writes alike. A kernel bound by one resource states that half alone with
/// [`compute_bound`] or [`memory_bound`].
pub fn roofline_bounds(
    client: &Client,
    compute_key: ThroughputKey,
    work: Work,
    thresholds: Thresholds,
) -> Bounds {
    Bounds {
        bounds: alloc::vec![
            compute_bound(client, compute_key, work, thresholds.compute),
            memory_bound(client, MemoryAccess::Copy, work, thresholds.memory),
        ],
        launch_overhead: measure_launch_overhead(client),
    }
}

/// Asynchronously builds compute and memory roofline bounds.
pub async fn roofline_bounds_async(
    client: &Client,
    compute_key: ThroughputKey,
    work: Work,
    thresholds: Thresholds,
) -> Bounds {
    Bounds {
        bounds: alloc::vec![
            compute_bound_async(client, compute_key, work, thresholds.compute).await,
            memory_bound_async(client, MemoryAccess::Copy, work, thresholds.memory).await,
        ],
        launch_overhead: measure_launch_overhead_async(client).await,
    }
}

/// The compute half of a roofline: `work`'s operations against the peak of `compute_key`,
/// close enough to it past `threshold`.
pub fn compute_bound(
    client: &Client,
    compute_key: ThroughputKey,
    work: Work,
    threshold: f32,
) -> AutotuneBound {
    // An unmeasurable ceiling is zero, which `time_at_peak` declines.
    let peak = measure_peak_throughput(client, compute_key).unwrap_or(ThroughputValue::ZERO);
    AutotuneBound {
        resource: ResourceBound {
            amount: work.compute_ops,
            peak_per_s: peak.ops_per_s(),
        },
        threshold,
    }
}

/// Asynchronously builds a compute roofline bound.
pub async fn compute_bound_async(
    client: &Client,
    compute_key: ThroughputKey,
    work: Work,
    threshold: f32,
) -> AutotuneBound {
    // An unmeasurable ceiling is zero, which `time_at_peak` declines.
    let peak = measure_peak_throughput_async(client, compute_key)
        .await
        .unwrap_or(ThroughputValue::ZERO);
    AutotuneBound {
        resource: ResourceBound {
            amount: work.compute_ops,
            peak_per_s: peak.ops_per_s(),
        },
        threshold,
    }
}

/// The memory half of a roofline: `work`'s bytes against the ceiling in the direction
/// `access` names, at the work's own footprint.
///
/// The direction is the caller's because a read-dominated kernel exceeds a copy's
/// bandwidth, half of which is a direction it never uses. Bounding one by
/// [`MemoryAccess::Copy`] passes a candidate at a fraction of the bus for close to peak.
pub fn memory_bound(
    client: &Client,
    access: MemoryAccess,
    work: Work,
    threshold: f32,
) -> AutotuneBound {
    let footprint = (work.bytes as u64).min(working_set_cap(client, access));
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory(MemorySpec::new(access, sweep_size(footprint))),
    };
    let peak = measure_peak_throughput(client, memory_key).unwrap_or(ThroughputValue::ZERO);
    AutotuneBound {
        resource: ResourceBound {
            amount: work.bytes,
            peak_per_s: peak.bytes_per_s(&memory_key),
        },
        threshold,
    }
}

/// Asynchronously builds a memory roofline bound.
pub async fn memory_bound_async(
    client: &Client,
    access: MemoryAccess,
    work: Work,
    threshold: f32,
) -> AutotuneBound {
    // Past what the device will allocate the probe measures the cap regardless,
    // so capping the ask keeps one cache entry rather than one per kernel.
    let footprint = (work.bytes as u64).min(working_set_cap(client, access));
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory(MemorySpec::new(access, sweep_size(footprint))),
    };
    let peak = measure_peak_throughput_async(client, memory_key)
        .await
        .unwrap_or(ThroughputValue::ZERO);
    AutotuneBound {
        resource: ResourceBound {
            amount: work.bytes,
            peak_per_s: peak.bytes_per_s(&memory_key),
        },
        threshold,
    }
}

/// What one launch costs on top of its kernel, which a time limit allows for.
pub fn measure_launch_overhead(client: &Client) -> core::time::Duration {
    let launch_key = ThroughputKey {
        mode: ThroughputMode::Launch,
    };
    measure_peak_throughput(client, launch_key)
        .map(|value| value.duration_per_op())
        .unwrap_or_default()
}

/// Asynchronously measures the overhead of one launch.
pub async fn measure_launch_overhead_async(client: &Client) -> core::time::Duration {
    let launch_key = ThroughputKey {
        mode: ThroughputMode::Launch,
    };
    measure_peak_throughput_async(client, launch_key)
        .await
        .map(|value| value.duration_per_op())
        .unwrap_or_default()
}

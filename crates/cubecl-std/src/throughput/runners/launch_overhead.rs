use cubecl::prelude::*;
use cubecl_core as cubecl;

pub async fn build_kernel(
    client: &cubecl_runtime::client::Client,
    _key: cubecl_runtime::throughput::ThroughputKey,
    _config: super::super::LaunchConfig,
) -> cubecl_runtime::throughput::KernelConfig {
    let client = client.clone();
    let sample = alloc::boxed::Box::new(
        move |iterations: usize| -> cubecl_environment::future::DynFut<_> {
            let client = client.clone();
            Box::pin(async move {
                let input = client.empty(core::mem::size_of::<i32>());
                let output = client.empty(core::mem::size_of::<i32>());

                // Opened before the window and read only if the device times
                // nothing: the fallback has to span the same launches, so it
                // starts with them.
                let start = cubecl_common::profile::Instant::now();

                let profiled = client.profile(
                    || unsafe {
                        for _ in 0..iterations {
                            launch_overhead::launch_unchecked(
                                &client,
                                cubecl_core::CubeCount::new_single(),
                                cubecl_core::server::CubeDim::new_single(),
                                1,
                                cubecl_core::frontend::BufferArg::from_raw_parts(input.clone(), 1),
                                cubecl_core::frontend::BufferArg::from_raw_parts(output.clone(), 1),
                                cubecl_core::ir::ElemType::Int(cubecl_core::ir::IntKind::I32),
                            );
                        }
                    },
                    "launch_overhead",
                );

                // What the device made of the window, when it made anything of
                // it. Two ways it makes nothing, both seen on Metal and neither
                // a broken probe:
                //
                // * [`ProfileError::NotMeasured`] — the window resolved no
                //   timestamped pass at all, so there is no span to report.
                // * `None` — the pair came back unwritten or out of order, which
                //   `stop_profile` declines to pass off as a duration.
                //
                // Any other profile error is a window that did not run as asked,
                // and it is not this probe's to interpret either.
                let measured = match profiled {
                    Ok((_, duration)) => duration.into_future().await.map(|ticks| ticks.duration()),
                    Err(_) => None,
                };

                // The host clock over the same launches, which is what every
                // other probe in this module times with. A launch costs the
                // caller what it adds to the wall clock, and `iterations` of
                // them amortize the single sync this pays for, so the host span
                // is a measurement of the same thing rather than a stand-in for
                // one.
                //
                // The alternative is to have no number: the benchmarker takes a
                // `Duration` and nothing else, so a probe that cannot answer
                // used to panic here — on a device where the ceiling is fine and
                // only its timer is quiet, taking the autotune task (and the
                // session on it) with it. A window that genuinely did not run
                // reads as ~0, which is exactly the `unwrap_or_default` a failed
                // probe already resolves to upstream.
                match measured {
                    Some(duration) => duration,
                    None => {
                        let _ = client.sync().await;
                        start.elapsed()
                    }
                }
            })
        },
    );

    cubecl_runtime::throughput::KernelConfig {
        sample,
        ops_count: 1,
        min_iterations: 1,
    }
}

#[cube(launch_unchecked)]
pub fn launch_overhead<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS == 0 {
        output[0] = input[0];
    }
}

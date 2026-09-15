use cubecl::prelude::*;
use cubecl_core as cubecl;

pub fn build_kernel(
    client: &cubecl_runtime::client::Client,
    _key: cubecl_runtime::throughput::ThroughputKey,
    _config: super::super::LaunchConfig,
) -> cubecl_runtime::throughput::KernelConfig {
    let client = client.clone();
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let input = client.empty(core::mem::size_of::<i32>());
        let output = client.empty(core::mem::size_of::<i32>());

        let (_, duration) = client
            .profile(
                || {
                    let _real_run = cubecl_runtime::dry_run::RealRun::new();
                    for _ in 0..iterations {
                        unsafe {
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
                    }
                },
                "launch_overhead",
            )
            .expect("should succeed launch_overhead");

        // A window whose timestamp slots come back unwritten resolves to
        // nothing: seen on Metal for the first window of a stream while other
        // streams are profiling, both slots reading zero and the next window
        // measuring. That is a timer reading zero, not a failure: the
        // benchmarker grows a window whose timer reads zero, and a probe that
        // never measures reports no timing rather than a number.
        cubecl_core::future::block_on(duration.into_future())
            .map_or(core::time::Duration::ZERO, |ticks| ticks.duration())
    });

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

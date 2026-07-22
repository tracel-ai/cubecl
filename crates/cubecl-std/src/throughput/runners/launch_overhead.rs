use cubecl::prelude::*;
use cubecl_core as cubecl;

pub fn build_kernel<R: cubecl_runtime::runtime::Runtime>(
    client: &cubecl_runtime::client::ComputeClient<R>,
    _key: cubecl_runtime::throughput::ThroughputKey,
    _config: super::super::LaunchConfig,
) -> cubecl_runtime::throughput::KernelConfig {
    let client = client.clone();
    let sample = alloc::boxed::Box::new(move |iterations: usize| {
        let input = client.empty(core::mem::size_of::<i32>());
        let output = client.empty(core::mem::size_of::<i32>());

        let (_, duration) = client
            .profile(
                || unsafe {
                    for _ in 0..iterations {
                        launch_overhead::launch_unchecked::<R>(
                            &client,
                            cubecl_core::CubeCount::new_single(),
                            cubecl_core::server::CubeDim::new_single(),
                            1,
                            cubecl_core::frontend::BufferArg::from_raw_parts(input.clone(), 1),
                            cubecl_core::frontend::BufferArg::from_raw_parts(output.clone(), 1),
                            cubecl_core::ir::ElemType::Int(cubecl_core::ir::IntKind::I32).into(),
                        );
                    }
                },
                "launch_overhead",
            )
            .expect("should succeed launch_overhead");

        cubecl_core::future::block_on(duration.into_future()).duration()
    });

    cubecl_runtime::throughput::KernelConfig {
        sample,
        ops_count: 1,
    }
}

#[cube(launch_unchecked)]
pub fn launch_overhead<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    output: &mut [Vector<I, N>],
    #[define(I)] _dtype: StorageType,
) {
    if ABSOLUTE_POS == 0 {
        output[0] = input[0];
    }
}

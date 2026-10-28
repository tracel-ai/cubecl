use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cuda::CudaRuntime;
use cubecl_server::memory_management::{MemoryReport, MemoryScope};
use cubecl_server::runtime::Runtime;

#[cube(launch)]
fn add_one(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + f32::cast_from(input.shape(0));
    }
}

fn dynamic_allocs(report: &MemoryReport) -> u64 {
    report
        .streams
        .iter()
        .flat_map(|stream| &stream.pools.dynamic)
        .map(|pool| pool.usage.number_allocs)
        .sum()
}

#[test]
fn changing_shapes_do_not_pollute_tensor_pools_or_disable_cache_reuse() {
    let client = CudaRuntime::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&vec![1.0; 256]));
    let output = client.empty(256 * core::mem::size_of::<f32>());
    let launch = |n: usize| {
        add_one::launch(
            &client,
            CubeCount::Static(8, 1, 1),
            CubeDim::new(&client, 32),
            // SAFETY: input holds 256 contiguous f32 values, and n never exceeds 256.
            unsafe { TensorArg::from_raw_parts(input.clone(), [1].into(), [n].into()) },
            // SAFETY: output holds 256 contiguous f32 values, and n never exceeds 256.
            unsafe { TensorArg::from_raw_parts(output.clone(), [1].into(), [n].into()) },
        );
    };

    launch(256);
    client.read_one(output.clone()).unwrap();
    let baseline = client.memory_report(MemoryScope::Device);
    let baseline_allocs = baseline.usage().number_allocs;
    let baseline_dynamic = dynamic_allocs(&baseline);

    for n in 1..256 {
        launch(n);
    }
    let out = client.read_one(output.clone()).unwrap();
    let values = f32::from_bytes(&out);
    assert_eq!(&values[..255], &[256.0; 255]);
    assert_eq!(values[255], 257.0);
    assert_eq!(
        client
            .memory_report(MemoryScope::Device)
            .usage()
            .number_allocs,
        baseline_allocs,
        "single-use shapes must not retain metadata buffers"
    );

    launch(1);
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out)[0], 2.0);
    for n in 1..256 {
        launch(n);
    }
    let out = client.read_one(output.clone()).unwrap();
    let values = f32::from_bytes(&out);
    assert_eq!(&values[..255], &[256.0; 255]);
    assert_eq!(values[255], 257.0);
    let cached = client.memory_report(MemoryScope::Device);
    assert_eq!(
        cached.usage().number_allocs,
        baseline_allocs + 255,
        "repeated shapes must retain their metadata slots"
    );
    let metadata: Vec<_> = cached
        .streams
        .iter()
        .filter_map(|stream| stream.pools.metadata.as_ref())
        .collect();
    assert_eq!(metadata.iter().map(|pool| pool.pages).sum::<u64>(), 1);
    assert_eq!(
        metadata
            .iter()
            .map(|pool| pool.usage.number_allocs)
            .sum::<u64>(),
        255
    );
    assert_eq!(
        dynamic_allocs(&cached),
        baseline_dynamic,
        "retained metadata must not add live slices to tensor pools: {cached:?}"
    );

    launch(1);
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out)[0], 2.0);
    for n in 1..256 {
        launch(n);
    }
    launch(32);
    let out = client.read_one(output.clone()).unwrap();
    let values = f32::from_bytes(&out);
    assert_eq!(&values[..32], &[33.0; 32]);
    assert_eq!(values[32], 256.0);
    assert_eq!(values[255], 257.0);
    assert_eq!(client.memory_report(MemoryScope::Device), cached);

    let tensor = client.empty(64);
    assert_eq!(
        dynamic_allocs(&client.memory_report(MemoryScope::Device)),
        baseline_dynamic + 1
    );
    drop(tensor);

    let mut shape = vec![1; 129];
    shape[0] = 2;
    let strides = vec![1; 129];
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 32),
        // SAFETY: the rank-129 view addresses only the first two values.
        unsafe {
            TensorArg::from_raw_parts(input.clone(), strides.clone().into(), shape.clone().into())
        },
        // SAFETY: the rank-129 view addresses only the first two values.
        unsafe { TensorArg::from_raw_parts(output.clone(), strides.into(), shape.into()) },
    );
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(&f32::from_bytes(&out)[..3], &[3.0, 3.0, 33.0]);
    let report = client.memory_report(MemoryScope::Device);
    assert_eq!(dynamic_allocs(&report), baseline_dynamic);
    assert!(
        report
            .streams
            .iter()
            .filter_map(|stream| stream.pools.metadata.as_ref())
            .any(|pool| pool.largest_alloc > 2048)
    );
}

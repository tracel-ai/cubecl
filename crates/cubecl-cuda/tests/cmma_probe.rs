//! The cmma throughput probe's kernel writes its whole tile: every element of a
//! product of ones over `k`, `n_iter` times, is `k · n_iter`.

use cubecl_core as cubecl;
use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::*;
use cubecl_core::throughput::CmmaDims;
use cubecl_cuda::CudaRuntime;
use cubecl_server::runtime::Runtime;
use cubecl_std::throughput::compute_cmma::compute_cmma_throughput;

#[test]
fn the_cmma_probe_stores_its_whole_tile() {
    let device = Default::default();
    let client = CudaRuntime::client(&device);
    let dims = CmmaDims {
        m: 16,
        n: 16,
        k: 16,
    };
    let iterations = 4usize;
    let out = client.empty(dims.m * dims.n * size_of::<f32>());
    unsafe {
        compute_cmma_throughput::launch_unchecked(
            &client,
            CubeCount::Static(4, 1, 1),
            CubeDim::new(&client, 256),
            8,
            BufferArg::from_raw_parts(out.clone(), 1),
            iterations,
            dims,
            ElemType::Float(FloatKind::F16),
            ElemType::Float(FloatKind::F32),
        )
    };
    let bytes = client.read_one(out).expect("the tile reads back");
    let want = (dims.k * iterations) as f32;
    for (at, got) in f32::from_bytes(&bytes).iter().enumerate() {
        assert_eq!(*got, want, "element {at}");
    }
}

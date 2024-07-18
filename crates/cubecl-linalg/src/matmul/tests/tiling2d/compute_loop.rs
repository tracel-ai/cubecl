use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tiling2d::outer_product::tile_outer_product;
use crate::matmul::{
    tests::test_utils::{
        assert_equals, create_empty, make_tiling2d_config, range_tensor, range_tensor_transposed,
    },
    tiling2d::{
        base::{Coordinates, CoordinatesExpand, TILE_SIZE},
        compute_loop::compute_loop,
        config::CubeTiling2dConfig,
    },
};

#[cube(launch)]
#[allow(unused_mut)]
fn tile_outer_product_test<F: Float>(
    register_m: Array<F>,
    register_n: Array<F>,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    // We launch with array then convert to vectorized float,
    // because direct launch of vectorized float is not supported
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let register_m = register_m.to_vectorized(tile_size);
    let register_n = register_n.to_vectorized(tile_size);

    for i in range(
        0u32,
        Comptime::get(tile_size * tile_size),
        Comptime::new(false),
    ) {
        results[i] = F::new(0.);
    }
    tile_outer_product::<F>(register_m, register_n, results, config)
}

/// Exported test
pub fn tile_outer_product_vectorized_unit_test_2<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let register_m = client.create(f32::as_bytes(&[16., 20., 24., 28.]));
    let register_n = client.create(f32::as_bytes(&[4., 5., 6., 7.]));
    let results = create_empty::<R>(4, 4, device);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    tile_outer_product_test::launch::<F32, R>(
        client.clone(),
        cube_count,
        cube_dim,
        ArrayArg::new(&register_m, 4),
        ArrayArg::new(&register_n, 4),
        ArrayArg::new(&results, 16),
        config,
    );

    let expected = &[
        64.0, 80.0, 96.0, 112.0, 80.0, 100.0, 120.0, 140.0, 96.0, 120.0, 144.0, 168.0, 112.0,
        140.0, 168.0, 196.0,
    ];
    assert_equals::<R>(results, expected, device);
}

#[cube(launch)]
fn compute_loop_test<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    results: &mut Array<F>,
    lhs_len: Comptime<UInt>,
    rhs_len: Comptime<UInt>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_m);
    let sm_size_lhs = block_size_m * block_size_k / tile_size;
    let sm_size_rhs = block_size_n * block_size_k / tile_size;

    // Shared memories are not launchable, so we launch with tensor and convert to shared memory
    let mut shared_lhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_lhs), Comptime::get(tile_size));
    for i in range(0u32, Comptime::get(lhs_len), Comptime::new(true)) {
        shared_lhs[i] = lhs[i];
    }

    let mut shared_rhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_rhs), Comptime::get(tile_size));
    for i in range(0u32, Comptime::get(rhs_len), Comptime::new(true)) {
        shared_rhs[i] = rhs[i];
    }

    for i in range(0u32, 16u32, Comptime::new(false)) {
        results[i] = F::new(0.);
    }

    let coordinates = Coordinates {
        unit_row,
        unit_col,
        skip_row: UInt::new(0),
        skip_col: UInt::new(0),
    };

    compute_loop(coordinates, shared_lhs, shared_rhs, results, config)
}

/// Exported test
pub fn tile_outer_product_vectorized_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let register_m = client.create(f32::as_bytes(&[0., 1., 2., 3.]));
    let register_n = client.create(f32::as_bytes(&[1., 2., 3., 4.]));
    let results = create_empty::<R>(4, 4, device);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    tile_outer_product_test::launch::<F32, R>(
        client.clone(),
        cube_count,
        cube_dim,
        ArrayArg::new(&register_m, 4),
        ArrayArg::new(&register_n, 4),
        ArrayArg::new(&results, 16),
        config,
    );

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0,
    ];
    assert_equals::<R>(results, expected, device);
}

/// Exported test
pub fn compute_loop_unit_test<R: Runtime>(device: &R::Device) {
    let lhs = range_tensor::<R>(8, 8, device);
    let rhs = range_tensor::<R>(8, 8, device);
    let results = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    compute_loop_test::launch::<F32, R>(
        R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape),
        ScalarArg::new(0),
        ScalarArg::new(0),
        ArrayArg::new(&results, 16),
        UInt::new(16),
        UInt::new(16),
        config,
    );

    let expected = &[
        8960.0, 9184.0, 9408.0, 9632.0, 9184.0, 9416.0, 9648.0, 9880.0, 9408.0, 9648.0, 9888.0,
        10128.0, 9632.0, 9880.0, 10128.0, 10376.0,
    ];
    assert_equals::<R>(results, expected, device);
}

/// Exported test
pub fn compute_loop_unit_offset_test<R: Runtime>(device: &R::Device) {
    let lhs = range_tensor_transposed::<R>(8, 4, device);
    let rhs = range_tensor::<R>(4, 8, device);
    let results = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(4, 8, 4);

    compute_loop_test::launch::<F32, R>(
        R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape),
        TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape),
        ScalarArg::new(4),
        ScalarArg::new(4),
        ArrayArg::new(&results, 16),
        UInt::new(8),
        UInt::new(8),
        config,
    );

    let expected = &[
        1160.0, 1230.0, 1300.0, 1370.0, 1416.0, 1502.0, 1588.0, 1674.0, 1672.0, 1774.0, 1876.0,
        1978.0, 1928.0, 2046.0, 2164.0, 2282.0,
    ];
    assert_equals::<R>(results, expected, device);
}

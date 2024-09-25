use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tiling2d::outer_product::tile_outer_product;
use crate::matmul::{
    tests::test_utils::{assert_equals, create_empty, make_tiling2d_config, range_tensor},
    tiling2d::{
        base::{Coordinates, TILE_SIZE},
        compute_loop::compute_loop,
        config::CubeTiling2dConfig,
    },
};

#[cube(launch_unchecked)]
#[allow(unused_mut)]
fn tile_outer_product_test<F: Float>(
    register_m: Array<Line<F>>,
    register_n: Array<Line<F>>,
    results: &mut Array<F>,
    #[comptime] config: CubeTiling2dConfig,
) {
    // We launch with array then convert to vectorized float,
    // because direct launch of vectorized float is not supported
    let tile_size = config.tile_size;
    let register_m = register_m.to_vectorized(tile_size);
    let register_n = register_n.to_vectorized(tile_size);

    for i in 0..tile_size * tile_size {
        results[i] = F::new(0.);
    }
    tile_outer_product::<F>(register_m, register_n, results, config)
}

/// Exported test
pub fn tile_outer_product_vectorized_unit_test_2<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let register_m = client.create(f32::as_bytes(&[16., 20., 24., 28.]));
    let register_n = client.create(f32::as_bytes(&[4., 5., 6., 7.]));
    let results = create_empty::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    unsafe {
        tile_outer_product_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&register_m, 4, 1),
            ArrayArg::from_raw_parts(&register_n, 4, 1),
            ArrayArg::from_raw_parts(&results, 16, 1),
            config,
        );
    };

    let expected = &[
        64.0, 80.0, 96.0, 112.0, 80.0, 100.0, 120.0, 140.0, 96.0, 120.0, 144.0, 168.0, 112.0,
        140.0, 168.0, 196.0,
    ];
    assert_equals::<R>(&client, results, expected);
}

#[cube(launch_unchecked)]
fn compute_loop_test<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    unit_row: u32,
    unit_col: u32,
    results: &mut Array<F>,
    #[comptime] lhs_len: u32,
    #[comptime] rhs_len: u32,
    #[comptime] config: CubeTiling2dConfig,
) {
    let tile_size = config.tile_size;
    let block_size_m = config.block_size_m;
    let block_size_k = config.block_size_m;
    let block_size_n = config.block_size_m;
    let sm_size_lhs = block_size_m * block_size_k / tile_size;
    let sm_size_rhs = block_size_n * block_size_k / tile_size;

    // Shared memories are not launchable, so we launch with tensor and convert to shared memory
    let mut shared_lhs = SharedMemory::<F>::new_lined(sm_size_lhs, tile_size);
    #[unroll]
    for i in 0..lhs_len {
        shared_lhs[i] = lhs[i];
    }

    let mut shared_rhs = SharedMemory::<F>::new_lined(sm_size_rhs, tile_size);
    #[unroll]
    for i in 0..rhs_len {
        shared_rhs[i] = rhs[i];
    }

    for i in 0..16 {
        results[i] = F::new(0.);
    }

    let coordinates = Coordinates {
        unit_row,
        unit_col,
        skip_row: 0,
        skip_col: 0,
    };

    compute_loop(coordinates, shared_lhs, shared_rhs, results, config)
}

/// Exported test
pub fn tile_outer_product_vectorized_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let register_m = client.create(f32::as_bytes(&[0., 1., 2., 3.]));
    let register_n = client.create(f32::as_bytes(&[1., 2., 3., 4.]));
    let results = create_empty::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    unsafe {
        tile_outer_product_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&register_m, 4, 1),
            ArrayArg::from_raw_parts(&register_n, 4, 1),
            ArrayArg::from_raw_parts(&results, 16, 1),
            config,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0,
    ];
    assert_equals::<R>(&client, results, expected);
}

/// Exported test
pub fn compute_loop_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, 8, 8);
    let rhs = range_tensor::<R>(&client, 8, 8);
    let results = create_empty::<R>(&client, TILE_SIZE, TILE_SIZE);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    const SOME_DIM: usize = 12;
    let config = make_tiling2d_config(SOME_DIM, SOME_DIM, SOME_DIM);

    unsafe {
        compute_loop_test::launch_unchecked::<f32, R>(
            &R::client(device),
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, TILE_SIZE as u8),
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, TILE_SIZE as u8),
            ScalarArg::new(0),
            ScalarArg::new(0),
            ArrayArg::from_raw_parts(&results, 16, 1),
            16,
            16,
            config,
        );
    };

    let expected = &[
        8960.0, 9184.0, 9408.0, 9632.0, 9184.0, 9416.0, 9648.0, 9880.0, 9408.0, 9648.0, 9888.0,
        10128.0, 9632.0, 9880.0, 10128.0, 10376.0,
    ];
    assert_equals::<R>(&client, results, expected);
}

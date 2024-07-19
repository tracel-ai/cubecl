use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tests::test_utils::{
    make_tiling2d_config, range_tensor_transposed, zeros_tensor,
};
use crate::matmul::tiling2d::tile::writer::TileWriter;
use crate::matmul::tiling2d::write_output::write_to_output;
use crate::matmul::{
    tests::test_utils::{assert_equals, range_tensor},
    tiling2d::{
        base::{Coordinates, CoordinatesExpand, Dimensions, DimensionsExpand, TILE_SIZE},
        config::CubeTiling2dConfig,
    },
};

#[cube(launch)]
fn write_to_output_test<F: Float>(
    out: &mut Tensor<F>,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let coordinates = Coordinates {
        unit_row: UInt::new(4),
        unit_col: UInt::new(4),
        skip_row: UInt::new(0),
        skip_col: UInt::new(0),
    };
    let dims = Dimensions {
        m: out.shape(out.rank() - UInt::new(2)),
        k: UInt::new(0),
        n: out.shape(out.rank() - UInt::new(1)),
    };

    write_to_output::<F, TileWriter<F>>(out, results, coordinates, UInt::new(0), dims, config);
}

#[cube(launch)]
fn write_results_to_output_out_of_bounds_test<F: Float>(
    out: &mut Tensor<F>,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let coordinates = Coordinates {
        unit_row: UNIT_POS_X * UInt::new(4),
        unit_col: UNIT_POS_Y * UInt::new(4),
        skip_row: UInt::new(0),
        skip_col: UInt::new(0),
    };
    let dims = Dimensions {
        m: out.shape(out.rank() - UInt::new(2)),
        k: UInt::new(0),
        n: out.shape(out.rank() - UInt::new(1)),
    };

    write_to_output::<F, TileWriter<F>>(out, results, coordinates, UInt::new(0), dims, config);
}

/// Exported test
pub fn write_to_output_over_height_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let out = zeros_tensor::<R>(&client, 6, 8);
    let tile = range_tensor::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(6, 8, 8);

    write_to_output_test::launch::<F32, R>(
        &R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(TILE_SIZE as u8, &out.handle, &out.strides, &out.shape),
        ArrayArg::new(&tile.handle, 16),
        config,
    );

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0,
    ];
    assert_equals::<R>(&client, out.handle, expected);
}

/// Exported test
pub fn write_to_output_over_width_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let out = zeros_tensor::<R>(&client, 8, 4);
    let tile = range_tensor::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 4);

    write_to_output_test::launch::<F32, R>(
        &R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(TILE_SIZE as u8, &out.handle, &out.strides, &out.shape),
        ArrayArg::new(&tile.handle, 16),
        config,
    );

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_equals::<R>(&client, out.handle, expected);
}

/// Exported test
pub fn write_to_output_vectorized_less_than_tile_unit_test<R: Runtime>(device: &R::Device) {
    let vectorization = 2;
    let client = R::client(device);
    let out = zeros_tensor::<R>(&client, 8, 8);
    let tile = range_tensor::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 8);

    write_to_output_test::launch::<F32, R>(
        &R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(vectorization as u8, &out.handle, &out.strides, &out.shape),
        ArrayArg::new(&tile.handle, 16),
        config,
    );

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0,
        10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
    ];
    assert_equals::<R>(&client, out.handle, expected);
}

/// Exported test
pub fn write_to_output_scalar_unit_test<R: Runtime>(device: &R::Device) {
    let vectorization = 1;
    let client = R::client(device);
    let out = zeros_tensor::<R>(&client, 8, 8);
    let tile = range_tensor::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 8);

    write_to_output_test::launch::<F32, R>(
        &R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(vectorization as u8, &out.handle, &out.strides, &out.shape),
        ArrayArg::new(&tile.handle, 16),
        config,
    );

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0,
        10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
    ];
    assert_equals::<R>(&client, out.handle, expected);
}

/// Exported test
pub fn write_to_output_scalar_out_of_bounds_cube_test<R: Runtime>(device: &R::Device) {
    let vectorization = 1;
    let client = R::client(device);
    let out = zeros_tensor::<R>(&client, 5, 1);
    let results = range_tensor_transposed::<R>(&client, 4, 4);
    let cube_dim = CubeDim::new(2, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(5, 8, 1);

    write_results_to_output_out_of_bounds_test::launch::<F32, R>(
        &R::client(device),
        cube_count,
        cube_dim,
        TensorArg::vectorized(vectorization, &out.handle, &out.strides, &out.shape),
        ArrayArg::new(&results.handle, 16),
        config,
    );

    let expected = &[0.0, 1.0, 2.0, 3.0, 0.0];
    assert_equals::<R>(&client, out.handle, expected);
}

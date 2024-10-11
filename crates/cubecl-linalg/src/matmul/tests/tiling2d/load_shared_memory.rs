use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tests::test_utils::make_tiling2d_config;
use crate::matmul::tiling2d::load_shared_memory::{
    load_lhs_plain, load_lhs_transposed, load_rhs_plain, load_rhs_transposed,
};
use crate::matmul::tiling2d::tile::loader::TileLoader;
use crate::matmul::{
    tests::test_utils::{assert_equals, create_empty, range_tensor},
    tiling2d::{
        base::{Coordinates, Dimensions, TILE_SIZE},
        config::CubeTiling2dConfig,
        load_shared_memory::LoadInfo,
    },
};

#[cube(launch_unchecked, create_dummy_kernel)]
pub fn load_tensor_test<F: Float>(
    tensor: &Tensor<Line<F>>,
    sm_out: &mut Array<Line<F>>,
    unit_row: u32,
    unit_col: u32,
    k: u32,
    #[comptime] config: CubeTiling2dConfig,
    #[comptime] is_lhs: bool,
) {
    let tile_size = config.tile_size;
    let block_size_k = config.block_size_k;
    let block_size_m = config.block_size_m;
    let sm_size = block_size_k * block_size_m / tile_size;
    let mut shared_memory = SharedMemory::<F>::new_lined(sm_size, tile_size);

    for i in 0..sm_size {
        sm_out[i] = Line::empty(tile_size).fill(F::new(0.));
        shared_memory[i] = Line::empty(tile_size).fill(F::new(0.));
    }

    let batch_offset = 0;

    let coordinates = Coordinates {
        unit_row,
        unit_col,
        skip_row: 0,
        skip_col: 0,
    };

    if is_lhs {
        let dims = Dimensions {
            m: tensor.shape(tensor.rank() - 2),
            k: tensor.shape(tensor.rank() - 1),
            n: 0,
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_lhs_transposed::<F, TileLoader<F>>(tensor, info, config);
    } else {
        let dims = Dimensions {
            m: 0,
            k: tensor.shape(tensor.rank() - 2),
            n: tensor.shape(tensor.rank() - 1),
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_rhs_plain::<F, TileLoader<F>>(tensor, info, config);
    }

    for i in 0..sm_size {
        sm_out[i] = shared_memory[i];
    }
}

#[cube(launch_unchecked, create_dummy_kernel)]
pub fn load_tensor_permuted_test<F: Float>(
    tensor: &Tensor<Line<F>>,
    sm_out: &mut Array<Line<F>>,
    unit_row: u32,
    unit_col: u32,
    k: u32,
    #[comptime] config: CubeTiling2dConfig,
    #[comptime] is_lhs: bool,
) {
    let tile_size = config.tile_size;
    let block_size_k = config.block_size_k;
    let block_size_m = config.block_size_m;
    let sm_size = block_size_k * block_size_m / tile_size;
    let mut shared_memory = SharedMemory::<F>::new_lined(sm_size, tile_size);

    for i in 0..sm_size {
        sm_out[i] = Line::empty(tile_size).fill(F::new(0.));
        shared_memory[i] = Line::empty(tile_size).fill(F::new(0.));
    }

    let batch_offset = 0;

    let coordinates = Coordinates {
        unit_row,
        unit_col,
        skip_row: 0,
        skip_col: 0,
    };

    if is_lhs {
        // Permuted
        let dims = Dimensions {
            m: tensor.shape(tensor.rank() - 1),
            k: tensor.shape(tensor.rank() - 2),
            n: 0,
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_lhs_plain::<F, TileLoader<F>>(tensor, info, config);
    } else {
        // Permuted
        let dims = Dimensions {
            m: 0,
            k: tensor.shape(tensor.rank() - 1),
            n: tensor.shape(tensor.rank() - 2),
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_rhs_transposed::<F, TileLoader<F>>(tensor, info, config);
    }

    for i in 0..sm_size {
        sm_out[i] = shared_memory[i];
    }
}

#[cube(launch_unchecked)]
fn load_tensor_multiple_tiles_test<F: Float>(
    tensor: &Tensor<Line<F>>,
    sm_out: &mut Array<Line<F>>,
    k: u32,
    #[comptime] config: CubeTiling2dConfig,
    #[comptime] is_lhs: bool,
) {
    let tile_size = config.tile_size;
    let block_size_k = config.block_size_k;
    let block_size_m = config.block_size_m;
    let sm_size = block_size_k * block_size_m / tile_size;
    let mut shared_memory = SharedMemory::<F>::new_lined(sm_size, tile_size);

    for i in 0..sm_size {
        sm_out[i] = Line::empty(tile_size).fill(F::new(0.));
        shared_memory[i] = Line::empty(tile_size).fill(F::new(0.));
    }

    sync_units();

    let unit_row = 4 * UNIT_POS_X;
    let unit_col = 4 * UNIT_POS_Y;
    let batch_offset = 0;

    let coordinates = Coordinates {
        unit_row,
        unit_col,
        skip_row: 0,
        skip_col: 0,
    };

    if is_lhs {
        let dims = Dimensions {
            m: tensor.shape(tensor.rank() - 2),
            k: tensor.shape(tensor.rank() - 1),
            n: 0,
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_lhs_transposed::<F, TileLoader<F>>(tensor, info, config);
    } else {
        let dims = Dimensions {
            m: 0,
            k: tensor.shape(tensor.rank() - 2),
            n: tensor.shape(tensor.rank() - 1),
        };
        let info = LoadInfo::<F> {
            coordinates,
            k,
            batch_offset,
            shared_memory,
            dims,
        };

        load_rhs_plain::<F, TileLoader<F>>(tensor, info, config);
    }

    sync_units();

    for i in 0..sm_size {
        sm_out[i] = shared_memory[i];
    }
}

/// Exported test
pub fn load_lhs_transposed_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, 16, 16);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(16, 16, 8);

    unsafe {
        load_tensor_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            true,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        76.0, 92.0, 108.0, 124.0, 0.0, 0.0, 0.0, 0.0, 77.0, 93.0, 109.0, 125.0, 0.0, 0.0, 0.0, 0.0,
        78.0, 94.0, 110.0, 126.0, 0.0, 0.0, 0.0, 0.0, 79.0, 95.0, 111.0, 127.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_lhs_transposed_out_of_bounds_cube_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let vectorization_factor = 1;
    let lhs = range_tensor::<R>(&client, 5, 1);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(2, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(5, 1, 1);

    unsafe {
        load_tensor_multiple_tiles_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(
                &lhs.handle,
                &lhs.strides,
                &lhs.shape,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(0),
            config,
            true,
        );
    };

    let expected = &[
        0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_lhs_transposed_cube_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, 8, 8);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(2, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 8);

    unsafe {
        load_tensor_multiple_tiles_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(0),
            config,
            true,
        );
    };

    let expected = &[
        0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 1.0, 9.0, 17.0, 25.0, 33.0, 41.0, 49.0, 57.0,
        2.0, 10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 3.0, 11.0, 19.0, 27.0, 35.0, 43.0, 51.0,
        59.0, 4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 5.0, 13.0, 21.0, 29.0, 37.0, 45.0,
        53.0, 61.0, 6.0, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 7.0, 15.0, 23.0, 31.0, 39.0,
        47.0, 55.0, 63.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_lhs_transposed_offset_cube_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, 8, 16);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(2, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 16);

    unsafe {
        load_tensor_multiple_tiles_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(8),
            config,
            true,
        );
    };

    let expected = &[
        8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 9.0, 25.0, 41.0, 57.0, 73.0, 89.0, 105.0,
        121.0, 10.0, 26.0, 42.0, 58.0, 74.0, 90.0, 106.0, 122.0, 11.0, 27.0, 43.0, 59.0, 75.0,
        91.0, 107.0, 123.0, 12.0, 28.0, 44.0, 60.0, 76.0, 92.0, 108.0, 124.0, 13.0, 29.0, 45.0,
        61.0, 77.0, 93.0, 109.0, 125.0, 14.0, 30.0, 46.0, 62.0, 78.0, 94.0, 110.0, 126.0, 15.0,
        31.0, 47.0, 63.0, 79.0, 95.0, 111.0, 127.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_rhs_plain_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let rhs = range_tensor::<R>(&client, 16, 16);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 16, 16);

    unsafe {
        load_tensor_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, TILE_SIZE as u8),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            false,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        196.0, 197.0, 198.0, 199.0, 0.0, 0.0, 0.0, 0.0, 212.0, 213.0, 214.0, 215.0, 0.0, 0.0, 0.0,
        0.0, 228.0, 229.0, 230.0, 231.0, 0.0, 0.0, 0.0, 0.0, 244.0, 245.0, 246.0, 247.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_rhs_plain_cube_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let rhs = range_tensor::<R>(&client, 8, 8);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(2, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, 8, 8);

    unsafe {
        load_tensor_multiple_tiles_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, TILE_SIZE as u8),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(0),
            config,
            false,
        );
    };

    let expected = &[
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
        47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0,
        62.0, 63.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_rhs_plain_cube_offset_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let rhs = range_tensor::<R>(&client, 16, 8);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(2, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(16, 16, 8);

    unsafe {
        load_tensor_multiple_tiles_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, TILE_SIZE as u8),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(8),
            config,
            false,
        );
    };

    let expected = &[
        64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0,
        79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0,
        94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
        108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
        121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_lhs_plain_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, 16, 16);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(16, 16, 8);

    unsafe {
        load_tensor_permuted_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            true,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        196.0, 197.0, 198.0, 199.0, 0.0, 0.0, 0.0, 0.0, 212.0, 213.0, 214.0, 215.0, 0.0, 0.0, 0.0,
        0.0, 228.0, 229.0, 230.0, 231.0, 0.0, 0.0, 0.0, 0.0, 244.0, 245.0, 246.0, 247.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_lhs_plain_out_of_bounds_unit_test<R: Runtime>(device: &R::Device) {
    let (m, k) = (6, 14);
    let client = R::client(device);
    let lhs = range_tensor::<R>(&client, k, m);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(m, k, 8);

    unsafe {
        load_tensor_permuted_test::launch_unchecked::<f32, R>(
            &R::client(device),
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&lhs.handle, &lhs.strides, &lhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            true,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        76.0, 77.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 82.0, 83.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_rhs_transposed_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let rhs = range_tensor::<R>(&client, 16, 16);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(16, 16, 8);

    unsafe {
        load_tensor_permuted_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            false,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        76.0, 92.0, 108.0, 124.0, 0.0, 0.0, 0.0, 0.0, 77.0, 93.0, 109.0, 125.0, 0.0, 0.0, 0.0, 0.0,
        78.0, 94.0, 110.0, 126.0, 0.0, 0.0, 0.0, 0.0, 79.0, 95.0, 111.0, 127.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

/// Exported test
pub fn load_rhs_transposed_out_of_bounds_unit_test<R: Runtime>(device: &R::Device) {
    let (k, n) = (14, 6);
    let client = R::client(device);
    let rhs = range_tensor::<R>(&client, n, k);
    let sm_out = create_empty::<R>(&client, 8, 8);
    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    let config = make_tiling2d_config(8, k, n);

    unsafe {
        load_tensor_permuted_test::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts(&rhs.handle, &rhs.strides, &rhs.shape, 1),
            ArrayArg::from_raw_parts(&sm_out, 64, TILE_SIZE as u8),
            ScalarArg::new(4),
            ScalarArg::new(4),
            ScalarArg::new(8),
            config,
            false,
        );
    };

    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        68.0, 82.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 69.0, 83.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_equals::<R>(&client, sm_out, expected);
}

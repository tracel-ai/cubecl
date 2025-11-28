#![allow(missing_docs)]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::primitives::{
    reduce_max_shuffle, reduce_min_shuffle, reduce_prod_shuffle, reduce_sum_shuffle,
};

/// Test kernel: Each warp sums its lane IDs using shuffle reduction
/// Expected: All 32 lanes in each warp should get 496 (sum of 0..31)
#[cube(launch)]
fn kernel_warp_sum_lanes<F: Float>(output: &mut Tensor<F>) {
    let lane_id = UNIT_POS_PLANE;
    let my_value: F = F::cast_from(lane_id);

    // Butterfly reduction - all lanes get the sum
    let sum: F = reduce_sum_shuffle::<F>(my_value);

    output[ABSOLUTE_POS] = sum;
}

/// Test kernel: Find max lane ID in each warp (should be 31)
#[cube(launch)]
fn kernel_warp_max_lanes<F: Float>(output: &mut Tensor<F>) {
    let lane_id = UNIT_POS_PLANE;
    let my_value: F = F::cast_from(lane_id);

    let max_val: F = reduce_max_shuffle::<F>(my_value);

    output[ABSOLUTE_POS] = max_val;
}

/// Test kernel: Find min lane ID in each warp (should be 0)
#[cube(launch)]
fn kernel_warp_min_lanes<F: Float>(output: &mut Tensor<F>) {
    let lane_id = UNIT_POS_PLANE;
    let my_value: F = F::cast_from(lane_id);

    let min_val: F = reduce_min_shuffle::<F>(my_value);

    output[ABSOLUTE_POS] = min_val;
}

/// Test kernel: Product of small values to avoid overflow
/// Each lane contributes (1.0 + lane_id / 100.0)
#[cube(launch)]
fn kernel_warp_prod<F: Float>(output: &mut Tensor<F>) {
    let lane_id = UNIT_POS_PLANE;
    let my_value: F = F::new(1.0) + F::cast_from(lane_id) / F::new(100.0);

    let prod: F = reduce_prod_shuffle::<F>(my_value);

    output[ABSOLUTE_POS] = prod;
}

/// Reduce a 32x32 matrix where each warp reduces its row
#[cube(launch)]
fn kernel_matrix_row_reduce<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let row = CUBE_POS_Y;
    let col = UNIT_POS_PLANE;

    let value: F = input[row * 32 + col];
    let row_sum: F = reduce_sum_shuffle::<F>(value);

    // Only lane 0 writes the result
    if col == 0 {
        output[row] = row_sum;
    }
}

/// Test warp sum reduction
pub fn test_warp_sum<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !supports_plane_ops(&client) {
        return; // Skip if no plane support
    }

    let output_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; 64])); // 2 warps

    unsafe {
        kernel_warp_sum_lanes::launch::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(64, 1, 1), // 2 warps of 32 threads
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1], &[64], 1),
        )
        .unwrap();
    }

    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Sum of 0..31 = 496
    let expected_sum = 496.0f32;

    for (i, &value) in output.iter().enumerate() {
        assert!(
            (value - expected_sum).abs() < 1e-3,
            "Warp sum failed at position {i}: got {value}, expected {expected_sum}"
        );
    }
}

/// Test warp max reduction
pub fn test_warp_max<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !supports_plane_ops(&client) {
        return;
    }

    let output_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; 64]));

    unsafe {
        kernel_warp_max_lanes::launch::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(64, 1, 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1], &[64], 1),
        )
        .unwrap();
    }

    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Max lane ID is 31
    for (i, &value) in output.iter().enumerate() {
        assert!(
            (value - 31.0).abs() < 1e-3,
            "Warp max failed at position {i}: got {value}, expected 31"
        );
    }
}

/// Test warp min reduction
pub fn test_warp_min<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !supports_plane_ops(&client) {
        return;
    }

    let output_handle = client.create_from_slice(f32::as_bytes(&vec![999.0f32; 64]));

    unsafe {
        kernel_warp_min_lanes::launch::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(64, 1, 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1], &[64], 1),
        )
        .unwrap();
    }

    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Min lane ID is 0
    for (i, &value) in output.iter().enumerate() {
        assert!(
            value.abs() < 1e-3,
            "Warp min failed at position {i}: got {value}, expected 0"
        );
    }
}

/// Test warp product reduction
pub fn test_warp_prod<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !supports_plane_ops(&client) {
        return;
    }

    let output_handle = client.create_from_slice(f32::as_bytes(&[0.0f32; 32]));

    unsafe {
        kernel_warp_prod::launch::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(32, 1, 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1], &[32], 1),
        )
        .unwrap();
    }

    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Calculate expected product: Π(1 + i/100) for i=0..31
    let mut expected = 1.0f32;
    for i in 0..32 {
        expected *= 1.0 + (i as f32) / 100.0;
    }

    for (i, &value) in output.iter().enumerate() {
        let rel_error = ((value - expected) / expected).abs();
        assert!(
            rel_error < 0.01, // 1% tolerance
            "Warp prod failed at position {i}: got {value}, expected {expected}, rel_error={rel_error}"
        );
    }
}

/// Reduce 32 rows of 32 elements each using warp shuffles
pub fn test_matrix_row_reduce<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !supports_plane_ops(&client) {
        return;
    }

    // Create a 32x32 matrix where matrix[i][j] = i * 32 + j
    let input_data: Vec<f32> = (0..1024).map(|x| x as f32).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input_data));
    let output_handle = client.create_from_slice(f32::as_bytes(&[0.0f32; 32]));

    unsafe {
        kernel_matrix_row_reduce::launch::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(32, 32, 1), // 32x32 = 1024 threads, 32 warps
            TensorArg::from_raw_parts::<f32>(&input_handle, &[1], &[1024], 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1], &[32], 1),
        )
        .unwrap();
    }

    let bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&bytes);

    // Row i should sum to: Σ(i*32 + j) for j=0..31 = i*32*32 + 496
    for (row, &value) in output.iter().enumerate() {
        let expected = (row as f32) * 32.0 * 32.0 + 496.0;
        assert!(
            (value - expected).abs() < 1e-2,
            "Matrix row reduce failed at row {row}: got {value}, expected {expected}"
        );
    }
}

fn supports_plane_ops<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client
        .properties()
        .features
        .plane
        .contains(cubecl_runtime::Plane::Ops)
}

use std::fmt::Debug;

use crate::{
    self as cubecl, Feature, TmaFeature,
    prelude::barrier::{Barrier, BarrierLevel},
};

use cubecl::prelude::*;
use cubecl_runtime::{server::ComputeServer, storage::ComputeStorage};

#[cube(launch)]
fn tensormap_load<F: Float>(input: &TensorMap<F>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::<F>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
    let mut stage = SharedMemory::<F>::new_aligned(32u32 * 16, 1u32, 128u32);

    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, &mut stage.to_slice_mut(), 0, 8);
        barrier.arrive_tx(1, 32 * 16 * F::elem_size());
    } else {
        barrier.arrive();
    }
    barrier.wait();

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos] = stage[out_pos];
}

#[cube(launch)]
fn tensormap_store<F: Float>(input: &Array<Line<F>>, output: &mut TensorMap<F>) {
    let mut shared = SharedMemory::new_aligned(32u32 * 16, 1u32, 128u32);

    let in_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    shared[in_pos] = input[in_pos];

    sync_proxy_shared();
    sync_cube();

    if UNIT_POS == 0 {
        tma_store_2d(&shared.to_slice(), output, 16, 8);
        tma_group_commit();
        tma_group_wait_read(0u32);
    }
}

#[cube(launch)]
fn tensormap_im2col_load<F: Float>(
    input: &TensorMap<F>,
    output: &mut Tensor<Line<F>>,
    #[comptime] tile_m: u32,
    #[comptime] kernel_h: u16,
    #[comptime] kernel_w: u16,
    #[comptime] channels: u32,
    #[comptime] pad_h: i32,
    #[comptime] pad_w: i32,
) {
    let tile_k = comptime!(kernel_h as u32 * kernel_w as u32);
    let tile_width = tile_m * channels; // Preserve 128-byte alignment, works for all float kinds.

    let barrier = Barrier::<F>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));
    let mut stage = SharedMemory::<F>::new_aligned(tile_k * tile_width, 1u32, 128u32);

    if UNIT_POS == 0 {
        #[unroll]
        for kernel_y in 0..kernel_h {
            #[unroll]
            for kernel_x in 0..kernel_w {
                let kernel_idx = kernel_y * kernel_w + kernel_x;
                let slice_start = kernel_idx as u32 * tile_width;
                let slice_end = slice_start + tile_width;
                let mut stage_slice = stage.slice_mut(slice_start, slice_end);
                barrier.tma_load_im2col_4d(
                    input,
                    &mut stage_slice,
                    0,
                    -pad_h,
                    -pad_w,
                    0,
                    kernel_y,
                    kernel_x,
                );
            }
        }
        barrier.arrive_tx(1, tile_width * tile_k * F::elem_size());
    } else {
        barrier.arrive();
    }
    barrier.wait();

    output[ABSOLUTE_POS] = stage[ABSOLUTE_POS];
}

#[cube(launch)]
fn tensormap_metadata<F: Float>(
    input_1: &Tensor<Line<F>>,
    output: &mut TensorMap<F>,
    input_2: &TensorMap<F>,
    output_2: &mut Tensor<u32>,
) {
    output_2[0] = input_1.shape(0);
    output_2[1] = input_2.shape(0);
    output_2[2] = output.shape(0);
    output_2[3] = output_2.shape(0);
}

pub fn test_tensormap_load<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..64 * 64).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let shape = vec![64, 64];
    let (handle, strides) = client.create_tensor(F::as_bytes(&values), &shape, size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, &strides, &shape, 1) };
    let out = client.empty(16 * 32 * size_of::<F>());

    tensormap_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 32],
            },
            input,
            F::as_elem_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts::<F>(&out, 32 * 16, 1) },
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);
    let expected: Vec<F> = (0..16)
        .flat_map(|i| i * 64..i * 64 + 32)
        .map(|it| F::from_int(it + 8))
        .collect();

    assert_eq!(actual, &expected);
}

pub fn test_tensormap_store<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..32 * 16).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let handle = client.create(F::as_bytes(&values));
    let (out, out_strides) = client.create_tensor(
        &vec![0u8; 64 * 64 * size_of::<F>()],
        &[64, 64],
        size_of::<F>(),
    );

    tensormap_store::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 32 * 16, 1) },
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 32],
            },
            unsafe { TensorArg::from_raw_parts::<F>(&out, &out_strides, &[64, 64], 1) },
            F::as_elem_native_unchecked(),
        ),
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);
    let mut expected: Vec<F> = vec![F::from_int(0); 64 * 64];
    for y in 0..16 {
        for x in 0..32 {
            let val = y * 32 + x;
            let y = y + 16;
            let x = x + 8;
            let index = y * 64 + x;
            expected[index] = F::from_int(val as i64);
        }
    }

    assert_eq!(actual, &expected);
}

pub fn test_tensormap_load_im2col<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let n = 1;
    let h = 3;
    let w = 3;
    let c = 8;

    let kernel_h = 2;
    let kernel_w = 2;

    let pad_h = 1;
    let pad_w = 1;
    let corner_h = pad_h - (kernel_h as i32 - 1);
    let corner_w = pad_w - (kernel_w as i32 - 1);

    let out_h = 4;
    let out_w = 4;

    let tile_m = n * out_h * out_w;
    let tile_k = kernel_h * kernel_w * c;
    let out_size = tile_m * tile_k;

    let values = (1..h * w * c + 1)
        .map(|it| F::from_int(it as i64))
        .collect::<Vec<_>>();
    let shape = [n, h, w, c];
    let (handle, strides) = client.create_tensor(F::as_bytes(&values), &shape, size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, &strides, &shape, 1) };
    let out_shape = [tile_k, tile_m];
    let out_strides = [tile_m, 1];
    let out = client.empty(out_size * size_of::<F>());

    tensormap_im2col_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(tile_m as u32 * c as u32, kernel_h as u32 * kernel_w as u32),
        TensorMapArg::new(
            TensorMapFormat::Im2col {
                pixel_box_lower_corner: vec![-pad_h, -pad_w],
                pixel_box_upper_corner: vec![corner_h, corner_w],
                channels_per_pixel: c as u32,
                pixels_per_column: tile_m as u32,
            },
            input,
            F::as_elem_native_unchecked(),
        ),
        unsafe { TensorArg::from_raw_parts::<F>(&out, &out_strides, &out_shape, 1) },
        tile_m as u32,
        kernel_h as u16,
        kernel_w as u16,
        c as u32,
        pad_h,
        pad_w,
    );

    let actual = client.read_one(out.binding());
    let actual = F::from_bytes(&actual);

    let mut expected = vec![0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9];
    expected.extend([0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]);
    expected.extend([0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0]);
    expected.extend([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0]);

    let expected_actual: Vec<F> = expected
        .iter()
        .flat_map(|v| {
            if *v == 0 {
                vec![0; c]
            } else {
                let ch_start = (*v - 1) * c + 1;
                (ch_start..ch_start + c).collect()
            }
        })
        .map(|v| F::from_int(v as i64))
        .collect();

    assert_eq!(actual, &expected_actual);
}

pub fn test_tensormap_metadata<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client
        .properties()
        .feature_enabled(Feature::Tma(TmaFeature::Base))
    {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let in_handle_1 = client.empty(4);
    let in_handle_2 = client.empty(64);
    let out_handle_1 = client.empty(64);
    let out_handle_2 = client.empty(size_of::<u32>() * 4);
    let strides = vec![16, 1];
    let input_1 = unsafe { TensorArg::from_raw_parts::<F>(&in_handle_1, &strides, &[2, 3], 1) };
    let input_2 = unsafe { TensorArg::from_raw_parts::<F>(&in_handle_2, &strides, &[4, 5], 1) };
    let output_1 = unsafe { TensorArg::from_raw_parts::<F>(&out_handle_1, &strides, &[6, 7], 1) };
    let output_2 = unsafe { TensorArg::from_raw_parts::<u32>(&out_handle_2, &strides, &[8, 9], 1) };

    tensormap_metadata::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        input_1,
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 16],
            },
            output_1,
            F::as_elem_native_unchecked(),
        ),
        TensorMapArg::new(
            TensorMapFormat::Tiled {
                tile_size: vec![16, 32],
            },
            input_2,
            F::as_elem_native_unchecked(),
        ),
        output_2,
    );

    let actual = client.read_one(out_handle_2.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[2, 4, 6, 8]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tensormap {
    () => {
        use super::*;

        #[test]
        fn test_tensormap_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_tensormap_load_im2col() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load_im2col::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_tensormap_store() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_store::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_tensormap_metadata() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_metadata::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

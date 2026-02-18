use crate::{self as cubecl, prelude::barrier::Barrier};
use cubecl::prelude::*;
use cubecl_ir::features::Tma;
use cubecl_runtime::{
    server::{Allocation, ComputeServer, CopyDescriptor},
    storage::ComputeStorage,
};
use cubecl_zspace::{Shape, shape, strides};
use std::fmt::Debug;

#[cube(launch)]
fn tensormap_load<F: Float>(input: &TensorMap<F, Tiled>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = SharedMemory::<F>::new_aligned(32usize * 16, 1usize, 128usize);

    let type_size = F::type_size();
    let expected = select(UNIT_POS == 0, comptime![32 * 16 * type_size] as u32, 0);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, &mut stage.to_slice_mut(), 0, 8);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos as usize] = stage[out_pos as usize];
}

#[cube(launch)]
fn tensormap_store<F: Float>(input: &Array<Line<F>>, output: &mut TensorMap<F, Tiled>) {
    let mut shared = SharedMemory::new_aligned(32usize * 16, 1usize, 128usize);

    let in_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    shared[in_pos as usize] = input[in_pos as usize];

    sync_async_proxy_shared();
    sync_cube();

    if UNIT_POS == 0 {
        tma_store_2d(&shared.to_slice(), output, 16, 8);
        tma_group_commit();
        tma_group_wait_read(0u32);
    }
}

#[cube(launch)]
fn tensormap_im2col_load<F: Float>(
    input: &TensorMap<F, Im2col>,
    output: &mut Tensor<Line<F>>,
    #[comptime] tile_m: usize,
    #[comptime] kernel_h: u16,
    #[comptime] kernel_w: u16,
    #[comptime] channels: usize,
    #[comptime] pad_h: i32,
    #[comptime] pad_w: i32,
) {
    let tile_k = comptime!(kernel_h as usize * kernel_w as usize);
    let tile_width = tile_m * channels; // Preserve 128-byte alignment, works for all float kinds.

    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = SharedMemory::<F>::new_aligned(tile_k * tile_width, 1usize, 128usize);

    let type_size = F::type_size();
    let expected = select(
        UNIT_POS == 0,
        comptime![tile_width * tile_k * type_size] as u32,
        0,
    );
    if UNIT_POS == 0 {
        #[unroll]
        for kernel_y in 0..kernel_h {
            #[unroll]
            for kernel_x in 0..kernel_w {
                let kernel_idx = kernel_y * kernel_w + kernel_x;
                let slice_start = kernel_idx as usize * tile_width;
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
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    output[ABSOLUTE_POS] = stage[ABSOLUTE_POS];
}

#[cube(launch)]
fn tensormap_metadata<F: Float>(
    input_1: &Tensor<Line<F>>,
    output: &mut TensorMap<F, Tiled>,
    input_2: &TensorMap<F, Tiled>,
    output_2: &mut Tensor<u32>,
) {
    output_2[0] = input_1.shape(0) as u32;
    output_2[1] = input_2.shape(0) as u32;
    output_2[2] = output.shape(0) as u32;
    output_2[3] = output_2.shape(0) as u32;
}

pub fn test_tensormap_load<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>)
where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.properties().features.tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..64 * 64).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let shape = shape![64, 64];
    let Allocation { handle, strides } =
        client.create_tensor_from_slice(F::as_bytes(&values), shape.clone(), size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, strides, shape, 1) };
    let out = client.empty(16 * 32 * size_of::<F>());

    tensormap_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            input,
            F::as_type_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts::<F>(&out, 32 * 16, 1) },
    )
    .unwrap();

    let actual = client.read_one(out);
    let actual = F::from_bytes(&actual);
    let expected: Vec<F> = (0..16)
        .flat_map(|i| i * 64..i * 64 + 32)
        .map(|it| F::from_int(it + 8))
        .collect();

    assert_eq!(actual, &expected);
}

pub fn test_tensormap_store<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>)
where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.properties().features.tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..32 * 16).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let handle = client.create_from_slice(F::as_bytes(&values));
    let out_shape = &[64, 64];
    let out = client.create_tensor_from_slice(
        &vec![0u8; 64 * 64 * size_of::<F>()],
        out_shape.into(),
        size_of::<F>(),
    );

    tensormap_store::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 32 * 16, 1) },
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            unsafe {
                TensorArg::from_raw_parts::<F>(&out.handle, out.strides.clone(), [64, 64].into(), 1)
            },
            F::as_type_native_unchecked(),
        ),
    )
    .unwrap();

    let actual = client.read_one_tensor(CopyDescriptor::new(
        out.handle.binding(),
        out_shape.into(),
        out.strides.clone(),
        size_of::<F>(),
    ));
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

pub fn test_tensormap_load_im2col<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>)
where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.properties().features.tma.contains(Tma::Base) {
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
    let shape: Shape = [n, h, w, c].into();
    let Allocation { handle, strides } =
        client.create_tensor_from_slice(F::as_bytes(&values), shape.clone(), size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts::<F>(&handle, strides.into(), shape, 1) };
    let out_shape = [tile_k, tile_m];
    let out_strides = [tile_m, 1];
    let out = client.empty(out_size * size_of::<F>());

    tensormap_im2col_load::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(tile_m as u32 * c as u32, kernel_h as u32 * kernel_w as u32),
        TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: vec![-pad_h, -pad_w],
                pixel_box_upper_corner: vec![corner_h, corner_w],
                channels_per_pixel: c as u32,
                pixels_per_column: tile_m as u32,
            },
            input,
            F::as_type_native_unchecked(),
        ),
        unsafe { TensorArg::from_raw_parts::<F>(&out, out_strides.into(), out_shape.into(), 1) },
        tile_m,
        kernel_h as u16,
        kernel_w as u16,
        c,
        pad_h,
        pad_w,
    )
    .unwrap();

    let actual = client.read_one(out);
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

pub fn test_tensormap_metadata<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>)
where
    <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.properties().features.tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let in_handle_1 = client.empty(4);
    let in_handle_2 = client.empty(64);
    let out_handle_1 = client.empty(64);
    let out_handle_2 = client.empty(size_of::<u32>() * 4);
    let strides = strides![16, 1];
    let input_1 =
        unsafe { TensorArg::from_raw_parts::<F>(&in_handle_1, strides.clone(), [2, 3].into(), 1) };
    let input_2 =
        unsafe { TensorArg::from_raw_parts::<F>(&in_handle_2, strides.clone(), [4, 5].into(), 1) };
    let output_1 =
        unsafe { TensorArg::from_raw_parts::<F>(&out_handle_1, strides.clone(), [6, 7].into(), 1) };
    let output_2 =
        unsafe { TensorArg::from_raw_parts::<u32>(&out_handle_2, strides, [8, 9].into(), 1) };

    tensormap_metadata::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        input_1,
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 16],
            },
            output_1,
            F::as_type_native_unchecked(),
        ),
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            input_2,
            F::as_type_native_unchecked(),
        ),
        output_2,
    )
    .unwrap();

    let actual = client.read_one(out_handle_2);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[2, 4, 6, 8]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tensormap {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_tensormap_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_tensormap_load_im2col() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load_im2col::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_tensormap_store() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_store::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_tensormap_metadata() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_metadata::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

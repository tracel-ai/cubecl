use crate::{self as cubecl, prelude::barrier::Barrier};
use alloc::{fmt::Debug, vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_ir::features::Tma;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::{
    server::{CopyDescriptor, MemoryLayout, ServerStorage},
    storage::ComputeStorage,
};
use cubecl_zspace::{Shape, shape, strides};
use std::println;

#[cube(launch)]
fn tensormap_load<F: Float, N: Size>(input: &TensorMap<F, Tiled>, output: &mut [Vector<F, N>]) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage: Shared<[Vector<F, N>]> = Shared::new_aligned_slice(32usize * 16, 128usize);

    let type_size = F::size();
    let expected = select(UNIT_POS == 0, comptime![32 * 16 * type_size] as u32, 0);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, stage.as_mut_slice(), 0, 8);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos as usize] = stage[out_pos as usize];
}

#[cube(launch)]
fn tensormap_store<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut TensorMap<F, Tiled>) {
    let mut shared: Shared<[Vector<F, N>]> = Shared::new_aligned_slice(32usize * 16, 128usize);

    let in_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    shared[in_pos as usize] = input[in_pos as usize];

    sync_async_proxy_shared();
    sync_cube();

    if UNIT_POS == 0 {
        tma_store_2d(shared.as_slice(), output, 16, 8).wait_complete();
    }
}

/// One unit stores `stages` tiles of `16 x 32` through two shared memory buffers, each written
/// again once the store that read it two stages ago completed.
#[cube(launch)]
fn tensormap_store_pipeline<F: Float>(input: &[F], output: &mut TensorMap<F, Tiled>, stages: u32) {
    let mut buffers: Shared<[F]> = Shared::new_aligned_slice(2 * 512usize, 128usize);
    for i in 0..512usize {
        buffers[i] = input[i];
    }
    sync_async_proxy_shared();
    let mut previous = tma_store_2d(&buffers[0..512], output, 0, 0);
    for stage in 1..stages {
        let at = (stage % 2) as usize * 512;
        // The buffer was last read by the store two stages ago, which the last wait retired.
        for i in 0..512usize {
            buffers[at + i] = input[stage as usize * 512 + i];
        }
        sync_async_proxy_shared();
        let stored = tma_store_2d(&buffers[at..at + 512], output, stage as i32 * 16, 0);
        previous.wait();
        previous = stored;
    }
    previous.wait();
}

const F16_BYTES: usize = 2;

/// Loads `layout`'s tile with TMA, one load per panel, and copies it back out in the order of
/// `input`, finding each element through
/// [`WgmmaTileLayout::offset`](crate::wgmma::WgmmaTileLayout::offset). `input` is `lines x
/// contiguous`, with `lines` the tile's rows when it is K-major and its K when it is MN-major.
#[cube(launch)]
fn tensormap_load_swizzled(
    input: &TensorMap<half::f16, Tiled>,
    output: &mut [half::f16],
    #[comptime] layout: crate::wgmma::WgmmaTileLayout,
    #[comptime] lines: usize,
) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let size = comptime![layout.rows * layout.k];
    // TMA writes shared memory at 128-byte boundaries, past the 16 a tile without swizzle needs.
    let alignment = comptime![layout.alignment().max(128)];
    let mut tile: Shared<[half::f16]> = Shared::new_aligned_slice(size, alignment);

    let contiguous = comptime![size / lines];
    let panel = comptime![layout.panel_elems(F16_BYTES)];
    let panel_size = comptime![lines * panel];
    let expected = select(UNIT_POS == 0, comptime![(size * F16_BYTES) as u32], 0);
    if UNIT_POS == 0 {
        #[unroll]
        for p in 0..comptime![contiguous / panel] {
            let start = p * panel_size;
            barrier.tma_load_2d(
                input,
                &mut tile[start..start + panel_size],
                0,
                (p * panel) as i32,
            );
        }
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let k_major = comptime![layout.major == crate::wgmma::Major::K];
    #[unroll]
    for i in 0..comptime![size.div_ceil(128)] {
        let index = i * 128 + UNIT_POS as usize;
        if index < size {
            let line = index / contiguous;
            let along = index % contiguous;
            let (row, k) = if k_major {
                (line, along)
            } else {
                (along, line)
            };
            output[index] = tile[layout.offset(row, k, F16_BYTES)];
        }
    }
}

#[cube(launch)]
fn tensormap_im2col_load<F: Float, N: Size>(
    input: &TensorMap<F, Im2col>,
    output: &mut Tensor<Vector<F, N>>,
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
    let mut stage: Shared<[Vector<F, N>]> =
        Shared::new_aligned_slice(tile_k * tile_width, 128usize);

    let type_size = F::size();
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
                let stage_slice = &mut stage[slice_start..slice_end];
                barrier.tma_load_im2col_4d(
                    input,
                    stage_slice,
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
fn tensormap_metadata<F: Float, N: Size>(
    input_1: &Tensor<Vector<F, N>>,
    output: &mut TensorMap<F, Tiled>,
    input_2: &TensorMap<F, Tiled>,
    output_2: &mut Tensor<u32>,
) {
    output_2[0] = input_1.shape(0) as u32;
    output_2[1] = input_2.shape(0) as u32;
    output_2[2] = output.shape(0) as u32;
    output_2[3] = output_2.shape(0) as u32;
}

pub fn test_tensormap_load<R: Runtime, F: Float + CubeElement>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.features().tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let values = (0..64 * 64).map(|it| F::from_int(it)).collect::<Vec<_>>();
    let shape = shape![64, 64];
    let MemoryLayout {
        memory: handle,
        strides,
    } = client.create_tensor_from_slice(F::as_bytes(&values), shape.clone(), size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts(handle.clone(), strides, shape) };
    let out = client.empty(16 * 32 * size_of::<F>());

    tensormap_load::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        1,
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            input,
            F::elem_type_native(),
        ),
        unsafe { BufferArg::from_raw_parts(out.clone(), 32 * 16) },
    );

    let actual = client.read_one_unchecked(out);
    let actual = F::from_bytes(&actual);
    let expected: Vec<F> = (0..16)
        .flat_map(|i| i * 64..i * 64 + 32)
        .map(|it| F::from_int(it + 8))
        .collect();

    assert_eq!(actual, &expected);
}

pub fn test_tensormap_store<R: Runtime, F: Float + CubeElement>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.features().tma.contains(Tma::Base) {
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

    tensormap_store::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        1,
        unsafe { BufferArg::from_raw_parts(handle.clone(), 32 * 16) },
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            unsafe {
                TensorArg::from_raw_parts(out.memory.clone(), out.strides.clone(), [64, 64].into())
            },
            F::elem_type_native(),
        ),
    );

    let actual = client.read_one_unchecked_tensor(CopyDescriptor::new(
        out.memory.clone().binding(),
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

pub fn test_tensormap_store_pipeline<R: Runtime, F: Float + CubeElement>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.features().tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_store_pipeline due to unavailability");
        return;
    }

    let stages = 5;
    let values = (0..stages * 512)
        .map(|it| F::from_int(it as i64))
        .collect::<Vec<_>>();
    let handle = client.create_from_slice(F::as_bytes(&values));
    let out_shape = [stages * 16, 32];
    let out = client.create_tensor_from_slice(
        &vec![0u8; stages * 512 * size_of::<F>()],
        out_shape.into(),
        size_of::<F>(),
    );

    tensormap_store_pipeline::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), stages * 512) },
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            unsafe {
                TensorArg::from_raw_parts(out.memory.clone(), out.strides.clone(), out_shape.into())
            },
            F::elem_type_native(),
        ),
        stages as u32,
    );

    let actual = client.read_one_unchecked_tensor(CopyDescriptor::new(
        out.memory.clone().binding(),
        out_shape.into(),
        out.strides.clone(),
        size_of::<F>(),
    ));
    // Tile `s` is input elements `s * 512..`, stored at rows `s * 16..`: the output is the input.
    assert_eq!(F::from_bytes(&actual), &values);
}

pub fn test_tensormap_load_swizzled<R: Runtime>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    use crate::wgmma::{Major, Swizzle, WgmmaTileLayout};

    if !client.features().tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load_swizzled due to unavailability");
        return;
    }

    let cases = [
        (Major::K, Swizzle::None, 16, 16),
        (Major::K, Swizzle::B32, 16, 32),
        (Major::K, Swizzle::B64, 16, 64),
        (Major::K, Swizzle::B128, 16, 128),
        (Major::MN, Swizzle::None, 16, 16),
        (Major::MN, Swizzle::B32, 32, 16),
        (Major::MN, Swizzle::B64, 64, 16),
        (Major::MN, Swizzle::B128, 128, 16),
    ];
    for (major, swizzle, rows, k) in cases {
        let layout = WgmmaTileLayout {
            major,
            swizzle,
            rows,
            k,
        };
        layout.validate(F16_BYTES).expect("a layout an MMA reads");
        let (lines, contiguous) = match major {
            Major::K => (rows, k),
            Major::MN => (k, rows),
        };
        let size = rows * k;
        let values = (0..size)
            .map(|it| half::f16::from_f32(it as f32))
            .collect::<Vec<_>>();
        let shape: Shape = [lines, contiguous].into();
        let MemoryLayout {
            memory: handle,
            strides,
        } = client.create_tensor_from_slice(half::f16::as_bytes(&values), shape.clone(), F16_BYTES);
        let input = unsafe { TensorArg::from_raw_parts(handle.clone(), strides, shape) };
        let out = client.empty(size * F16_BYTES);

        tensormap_load_swizzled::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(128),
            TensorMapArg::new(
                TiledArgs {
                    tile_size: shape![lines, layout.panel_elems(F16_BYTES)],
                },
                input,
                half::f16::elem_type_native(),
            )
            .with_swizzle(layout.tensor_map_swizzle()),
            unsafe { BufferArg::from_raw_parts(out.clone(), size) },
            layout,
            lines,
        );

        let actual = client.read_one_unchecked(out);
        let actual = half::f16::from_bytes(&actual);
        assert_eq!(actual.len(), size, "the kernel did not run");
        assert_eq!(actual, &values, "{layout:?}");
    }
}

pub fn test_tensormap_load_im2col<R: Runtime, F: Float + CubeElement>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.features().tma.contains(Tma::Base) {
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
    let MemoryLayout {
        memory: handle,
        strides,
    } = client.create_tensor_from_slice(F::as_bytes(&values), shape.clone(), size_of::<F>());
    let input = unsafe { TensorArg::from_raw_parts(handle, strides, shape) };
    let out_shape = [tile_k, tile_m];
    let out_strides = [tile_m, 1];
    let out = client.empty(out_size * size_of::<F>());

    tensormap_im2col_load::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(tile_m as u32 * c as u32, kernel_h as u32 * kernel_w as u32),
        1,
        TensorMapArg::new(
            Im2colArgs {
                pixel_box_lower_corner: vec![-pad_h, -pad_w],
                pixel_box_upper_corner: vec![corner_h, corner_w],
                channels_per_pixel: c as u32,
                pixels_per_column: tile_m as u32,
            },
            input,
            F::elem_type_native(),
        ),
        unsafe { TensorArg::from_raw_parts(out.clone(), out_strides.into(), out_shape.into()) },
        tile_m,
        kernel_h as u16,
        kernel_w as u16,
        c,
        pad_h,
        pad_w,
    );

    let actual = client.read_one_unchecked(out);
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

pub fn test_tensormap_metadata<R: Runtime, F: Float + CubeElement>(client: Client)
where
    <<R::Server as ServerStorage>::Storage as ComputeStorage>::Resource: Debug,
{
    if !client.features().tma.contains(Tma::Base) {
        println!("Skipped test_tensormap_load due to unavailability");
        return;
    }

    let in_handle_1 = client.empty(4);
    let in_handle_2 = client.empty(64);
    let out_handle_1 = client.empty(64);
    let out_handle_2 = client.empty(size_of::<u32>() * 4);
    let strides = strides![16, 1];
    let input_1 = unsafe { TensorArg::from_raw_parts(in_handle_1, strides.clone(), [2, 3].into()) };
    let input_2 = unsafe { TensorArg::from_raw_parts(in_handle_2, strides.clone(), [4, 5].into()) };
    let output_1 =
        unsafe { TensorArg::from_raw_parts(out_handle_1.clone(), strides.clone(), [6, 7].into()) };
    let output_2 =
        unsafe { TensorArg::from_raw_parts(out_handle_2.clone(), strides, [8, 9].into()) };

    tensormap_metadata::launch::<F>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        1,
        input_1,
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 16],
            },
            output_1,
            F::elem_type_native(),
        ),
        TensorMapArg::new(
            TiledArgs {
                tile_size: shape![16, 32],
            },
            input_2,
            F::elem_type_native(),
        ),
        output_2,
    );

    let actual = client.read_one_unchecked(out_handle_2);
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
        fn test_tensormap_store_pipeline() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_store_pipeline::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_tensormap_load_swizzled() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensormap::test_tensormap_load_swizzled::<TestRuntime>(
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

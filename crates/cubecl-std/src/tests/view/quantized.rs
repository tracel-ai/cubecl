use cubecl::prelude::*;
use cubecl_common::{
    e2m1, e2m1x2,
    quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantValue},
};
use cubecl_core::{self as cubecl};
use half::f16;

use crate::tensor::{
    View,
    launch::ViewArg,
    layout::{plain::PlainLayout, *},
};

#[derive(CubeType, CubeLaunch)]
struct TestPerTensorScaleLayout {
    length: usize,
}

#[cube]
impl Layout for TestPerTensorScaleLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, _pos: Self::Coordinates) -> Self::SourceCoordinates {
        0
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), true)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
    }

    fn shape(&self) -> Self::Coordinates {
        self.length
    }
}

/// Which read method the kernel goes through. They differ only in how they handle a coordinate out
/// of bounds, so for the in-bounds coordinates these tests use they all owe the same value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadMode {
    Read,
    Checked,
    Masked,
    Unchecked,
}

#[cube(launch_unchecked)]
pub fn kernel_quantized_view<F: Float, N: Size>(
    lhs: View<'_, Vector<F, N>, Coords1d>,
    output: &mut [Vector<F, N>],
    #[comptime] mode: ReadMode,
) {
    let pos = UNIT_POS as usize;
    if pos < lhs.shape() {
        output[pos] = match mode {
            ReadMode::Read => lhs.read(pos),
            ReadMode::Checked => lhs.read_checked(pos),
            ReadMode::Masked => lhs.read_masked(pos, Vector::<F, N>::cast_from(F::new(0.0_f32))),
            ReadMode::Unchecked => lhs.read_unchecked(pos),
        };
    }
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_int<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    vector_size_values: VectorSize,
) {
    let vector_size_float = 8 * vector_size_values;

    let scheme = QuantScheme::default().with_value(QuantValue::Q4F);
    let float_data = (-8..=7)
        .map(|it| F::new(it as f32 * 3.4))
        .collect::<Vec<_>>();

    let output = client.empty(16 * size_of::<F>());
    let values = client.create_from_slice(u32::as_bytes(&[0xFEDCBA98, 0x76543210]));
    let scales = client.create_from_slice(f32::as_bytes(&[3.4]));

    let float_values = client.create_from_slice(F::as_bytes(&float_data));
    let float_output = client.empty(16 * size_of::<F>());

    let scales_layout = TestPerTensorScaleLayoutLaunch::new(16);

    let values_view =
        ViewArg::new_array::<PlainLayout>(unsafe { BufferArg::from_raw_parts(values, 2) }, ());
    let scales_view = ViewArg::new_array::<TestPerTensorScaleLayout>(
        unsafe { BufferArg::from_raw_parts(scales, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new_array::<PlainLayout>(
        unsafe { BufferArg::from_raw_parts(float_values, 16) },
        (),
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size_float,
            quantized_view,
            BufferArg::from_raw_parts(output.clone(), 16),
            ReadMode::Read,
        );
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size_float,
            float_view,
            BufferArg::from_raw_parts(float_output.clone(), 16),
            ReadMode::Read,
        );
    }

    let actual = client.read_one_unchecked(output);
    let actual_float = client.read_one_unchecked(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_fp4<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    vector_size_values: VectorSize,
) {
    if !client.properties().supports_type(e2m1x2::cube_type()) {
        return;
    }

    let vector_size_float = 8 * vector_size_values;

    let scheme = QuantScheme::default().with_value(QuantValue::E2M1);
    let float_data = (0..16)
        .map(e2m1::from_bits)
        .map(|it| F::new(it.to_f32() * 3.4))
        .collect::<Vec<_>>();

    let output = client.empty(16 * size_of::<F>());
    let values = client.create_from_slice(u32::as_bytes(&[0x76543210, 0xFEDCBA98]));
    let scales = client.create_from_slice(f32::as_bytes(&[3.4]));

    let float_values = client.create_from_slice(F::as_bytes(&float_data));
    let float_output = client.empty(16 * size_of::<F>());

    let scales_layout = TestPerTensorScaleLayoutLaunch::new(16);

    let values_view =
        ViewArg::new_array::<PlainLayout>(unsafe { BufferArg::from_raw_parts(values, 2) }, ());
    let scales_view = ViewArg::new_array::<TestPerTensorScaleLayout>(
        unsafe { BufferArg::from_raw_parts(scales, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new_array::<PlainLayout>(
        unsafe { BufferArg::from_raw_parts(float_values, 16) },
        (),
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size_float,
            quantized_view,
            BufferArg::from_raw_parts(output.clone(), 16),
            ReadMode::Read,
        );
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            vector_size_float,
            float_view,
            BufferArg::from_raw_parts(float_output.clone(), 16),
            ReadMode::Read,
        );
    }

    let actual = client.read_one_unchecked(output);
    let actual_float = client.read_one_unchecked(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

/// Two levels of scales: per-block scales normalized by one per-tensor scale.
///
/// Neither level reconstructs the values alone here: a block scale is far above the values it helps
/// rebuild, the per-tensor scale far below them. Every read method is exercised, since each one
/// pairs the per-tensor scale with its own read of the values and block scales.
///
/// Instantiated with a float narrower than the scales by
/// [`test_quantized_two_level_narrow_float`], which is what makes the block scales overflow `F`.
pub fn test_quantized_two_level_int<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    // One block per load, since the view assumes a single scale covers a whole read.
    let vector_size_float = 8;
    let block = 8;

    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q4F)
        .with_level(QuantLevel::block_tensor([block as u8], QuantParam::F32));

    // A power of two, so the reconstruction owes exactly the values the expectation computes.
    let global_scale = 2f32.powi(-20);
    let block_scales = [2f32.powi(18), 2f32.powi(19)];
    let expected = (0..16)
        .map(|i| F::new(global_scale * block_scales[i / block] * (i as f32 - 8.0)))
        .collect::<Vec<_>>();

    let values = client.create_from_slice(u32::as_bytes(&[0xFEDCBA98, 0x76543210]));
    let scales = client.create_from_slice(f32::as_bytes(&block_scales));
    let global = client.create_from_slice(f32::as_bytes(&[global_scale]));

    for mode in [
        ReadMode::Read,
        ReadMode::Checked,
        ReadMode::Masked,
        ReadMode::Unchecked,
    ] {
        let output = client.empty(16 * size_of::<F>());

        let values_view = ViewArg::new_array::<PlainLayout>(
            unsafe { BufferArg::from_raw_parts(values.clone(), 2) },
            (),
        );
        let scales_view = ViewArg::new_array::<PlainLayout>(
            unsafe { BufferArg::from_raw_parts(scales.clone(), 2) },
            (),
        );
        // The per-tensor scale is read from its first element, so it binds as a plain buffer.
        let global_buffer = unsafe { BufferArg::from_raw_parts(global.clone(), 1) };
        let quantized_view =
            ViewArg::new_quantized_two_level(values_view, scales_view, global_buffer, scheme);

        unsafe {
            kernel_quantized_view::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(2),
                vector_size_float,
                quantized_view,
                BufferArg::from_raw_parts(output.clone(), 16),
                mode,
            );
        }

        let actual = client.read_one_unchecked(output);
        let actual = F::from_bytes(&actual);

        assert_eq!(actual, &expected, "reading through {mode:?}");
    }
}

/// The per-tensor scale earns the f32 intermediate its keep here: the block scales overflow `f16`,
/// so folding the multiply back into `F` reconstructs every value as infinity.
///
/// Hardcodes the float type because the runtimes that run without a GPU instantiate the generic
/// tests with `f32`, which cannot observe that: both levels are stored as f32 to begin with, so a
/// multiply in `F` and a multiply in f32 are the same operation.
pub fn test_quantized_two_level_narrow_float<R: Runtime>(client: ComputeClient<R>) {
    if !client.properties().supports_type(f16::cube_type()) {
        return;
    }
    // The unroll pass cannot split the narrowing f32 -> f16 cast this test exists to exercise,
    // so a target whose native vectors are narrower than the vec8 loads dies at compile.
    if client.properties().hardware.max_vector_size < 8 {
        return;
    }

    test_quantized_two_level_int::<R, f16>(client);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quantized_view {
    ($ty: ty) => {
        use super::*;

        #[$crate::tests::test_log::test]
        fn test_quantized_view_per_tensor_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_std::tests::view::quantized::test_quantized_per_tensor_int::<TestRuntime, $ty>(
                client.clone(),
                1,
            );
            cubecl_std::tests::view::quantized::test_quantized_per_tensor_int::<TestRuntime, $ty>(
                client, 2,
            );
        }

        #[$crate::tests::test_log::test]
        fn test_quantized_view_per_tensor_fp4() {
            let client = TestRuntime::client(&Default::default());
            cubecl_std::tests::view::quantized::test_quantized_per_tensor_fp4::<TestRuntime, $ty>(
                client.clone(),
                1,
            );
            cubecl_std::tests::view::quantized::test_quantized_per_tensor_fp4::<TestRuntime, $ty>(
                client, 2,
            );
        }

        #[$crate::tests::test_log::test]
        fn test_quantized_view_two_level_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_std::tests::view::quantized::test_quantized_two_level_int::<TestRuntime, $ty>(
                client,
            );
        }

        #[$crate::tests::test_log::test]
        fn test_quantized_view_two_level_narrow_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_std::tests::view::quantized::test_quantized_two_level_narrow_float::<TestRuntime>(
                client,
            );
        }
    };
}

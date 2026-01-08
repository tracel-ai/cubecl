use cubecl::prelude::*;
use cubecl_common::{
    e2m1, e2m1x2,
    quant::scheme::{QuantScheme, QuantValue},
};
use cubecl_core::{self as cubecl};

use crate::tensor::{
    View,
    launch::ViewArg,
    layout::{
        plain::{PlainLayout, PlainLayoutLaunch},
        *,
    },
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
        0usize.runtime()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), true.runtime())
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        self.length
    }
}

#[cube(launch_unchecked)]
pub fn kernel_quantized_view<F: Float>(lhs: View<Line<F>, Coords1d>, output: &mut Array<Line<F>>) {
    if (UNIT_POS as usize) < lhs.shape() {
        output[UNIT_POS as usize] = lhs[UNIT_POS as usize];
    }
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_int<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    line_size_values: LineSize,
) {
    let line_size_float = 8 * line_size_values;
    let values_lines = 2 / line_size_values;

    let scheme = QuantScheme::default().with_value(QuantValue::Q4F);
    let float_data = (-8..=7)
        .map(|it| F::new(it as f32 * 3.4))
        .collect::<Vec<_>>();

    let output = client.empty(16 * size_of::<F>());
    let values = client.create_from_slice(u32::as_bytes(&[0xFEDCBA98, 0x76543210]));
    let scales = client.create_from_slice(f32::as_bytes(&[3.4]));

    let float_values = client.create_from_slice(F::as_bytes(&float_data));
    let float_output = client.empty(16 * size_of::<F>());

    let values_layout = PlainLayoutLaunch::new(ScalarArg::new(values_lines));
    let scales_layout = TestPerTensorScaleLayoutLaunch::new(ScalarArg::new(16));
    let float_layout = PlainLayoutLaunch::new(ScalarArg::new(values_lines));

    let values_view = ViewArg::new::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts::<u32>(&values, 2, line_size_values) },
        values_layout,
    );
    let scales_view = ViewArg::new::<TestPerTensorScaleLayout>(
        unsafe { ArrayArg::from_raw_parts::<f32>(&scales, 1, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts::<F>(&float_values, 16, line_size_float) },
        float_layout,
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            quantized_view,
            ArrayArg::from_raw_parts::<F>(&output, 16, line_size_float),
        )
        .unwrap();
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            float_view,
            ArrayArg::from_raw_parts::<F>(&float_output, 16, line_size_float),
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual_float = client.read_one(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

#[allow(clippy::needless_range_loop)]
pub fn test_quantized_per_tensor_fp4<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    line_size_values: LineSize,
) {
    if !client.properties().supports_type(e2m1x2::cube_type()) {
        return;
    }

    let line_size_float = 8 * line_size_values;
    let values_lines = 2 / line_size_values;

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

    let values_layout = PlainLayoutLaunch::new(ScalarArg::new(values_lines));
    let scales_layout = TestPerTensorScaleLayoutLaunch::new(ScalarArg::new(16));
    let float_layout = PlainLayoutLaunch::new(ScalarArg::new(values_lines));

    let values_view = ViewArg::new::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts::<u32>(&values, 2, line_size_values) },
        values_layout,
    );
    let scales_view = ViewArg::new::<TestPerTensorScaleLayout>(
        unsafe { ArrayArg::from_raw_parts::<f32>(&scales, 1, 1) },
        scales_layout,
    );
    let quantized_view = ViewArg::new_quantized(values_view, scales_view, scheme);
    let float_view = ViewArg::new::<PlainLayout>(
        unsafe { ArrayArg::from_raw_parts::<F>(&float_values, 16, line_size_float) },
        float_layout,
    );

    unsafe {
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            quantized_view,
            ArrayArg::from_raw_parts::<F>(&output, 16, line_size_float),
        )
        .unwrap();
        kernel_quantized_view::launch_unchecked::<F, R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(2),
            float_view,
            ArrayArg::from_raw_parts::<F>(&float_output, 16, line_size_float),
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual_float = client.read_one(float_output);
    let actual = F::from_bytes(&actual);
    let actual_float = F::from_bytes(&actual_float);

    assert_eq!(&actual, &float_data);
    assert_eq!(&actual_float, &float_data);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quantized_view {
    ($ty: ty) => {
        use super::*;

        #[test]
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

        #[test]
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
    };
}

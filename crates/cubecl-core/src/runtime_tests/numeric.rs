use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::{ElemType, FloatKind, UIntKind, features::TypeUsage};
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
pub fn kernel_define<N: Numeric>(array: &mut [N], #[define(N)] _elem: ElemType) {
    array[UNIT_POS as usize] += N::cast_from(5.0f32);
}

#[cube(launch)]
pub fn kernel_define_many<N: Numeric, N2: Numeric>(
    array: &mut [N],
    second: &[N2],
    #[define(N, N2)] _defines: [ElemType; 2],
) {
    array[UNIT_POS as usize] += N::cast_from(second[UNIT_POS as usize]);
}

pub fn test_kernel_define<R: Runtime>(client: Client) {
    let handle = client.create_from_slice(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));

    let elem = ElemType::Float(FloatKind::F32);

    kernel_define::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
        elem,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(6.0));
}

pub fn test_kernel_define_many<R: Runtime>(client: Client) {
    let first = client.create_from_slice(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));
    let second = client.create_from_slice(u32::as_bytes(&[u32::new(5), u32::new(6)]));

    let elem_first = ElemType::Float(FloatKind::F32);
    let elem_second = ElemType::UInt(UIntKind::U32);

    kernel_define_many::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(first.clone(), 2) },
        unsafe { BufferArg::from_raw_parts(second.clone(), 2) },
        [elem_first, elem_second],
    );

    let actual = client.read_one_unchecked(first);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(7.0));
}

type Tf32 = tf32;

#[cube(launch)]
pub fn kernel_tf32_round_scalar(array: &mut [f32]) {
    if ABSOLUTE_POS < array.len() {
        let rounded = Tf32::cast_from(array[ABSOLUTE_POS]);
        array[ABSOLUTE_POS] = f32::cast_from(rounded);
    }
}

#[cube(launch)]
pub fn kernel_tf32_round_vector(array: &mut [Vector<f32, Const<4>>]) {
    if ABSOLUTE_POS < array.len() {
        let rounded = Vector::<tf32, Const<4>>::cast_from(array[ABSOLUTE_POS]);
        array[ABSOLUTE_POS] = Vector::<f32, Const<4>>::cast_from(rounded);
    }
}

pub fn test_tf32_scalar_and_vector_rounding<R: Runtime>(client: Client) {
    if !tf32::supported_uses(&client).contains(TypeUsage::Conversion) {
        return;
    }

    // TF32 has 10 fraction bits: its spacing at 1.0 is 1/1024.
    let step = 1.0f32 / 1024.0;
    let tie = 1.0 + step / 2.0;
    let cases = [
        (tie - f32::EPSILON, 1.0),
        (tie, 1.0 + step),
        (tie + f32::EPSILON, 1.0 + step),
        (-tie, -(1.0 + step)),
    ];
    let values = cases.map(|(input, _)| input);
    let expected = cases.map(|(_, expected)| expected);
    let launches = [
        kernel_tf32_round_scalar::launch,
        kernel_tf32_round_vector::launch,
    ];
    for (lanes, launch) in [1, 4].into_iter().zip(launches) {
        let handle = client.create_from_slice(f32::as_bytes(&values));
        launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(32),
            unsafe { BufferArg::from_raw_parts(handle.clone(), values.len() / lanes) },
        );
        let bytes = client.read_one(handle).unwrap();
        assert_eq!(f32::from_bytes(&bytes), expected, "width={lanes}");
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_numeric {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_kernel_define() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::numeric::test_kernel_define::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_kernel_define_many() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::numeric::test_kernel_define_many::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_tf32_scalar_and_vector_rounding() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::numeric::test_tf32_scalar_and_vector_rounding::<TestRuntime>(client);
        }
    };
}

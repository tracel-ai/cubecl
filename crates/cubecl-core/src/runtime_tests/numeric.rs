use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::{ElemType, FloatKind, UIntKind};

#[cube(launch)]
pub fn kernel_define<N: Numeric>(array: &mut Array<N>, #[define(N)] _elem: ElemType) {
    array[UNIT_POS] += N::cast_from(5.0f32);
}

#[cube(launch)]
pub fn kernel_define_many<N: Numeric, N2: Numeric>(
    array: &mut Array<N>,
    second: Array<N2>,
    #[define(N, N2)] _defines: [ElemType; 2],
) {
    array[UNIT_POS] += N::cast_from(second[UNIT_POS]);
}

pub fn test_kernel_define<R: Runtime>(client: ComputeClient<R::Server>) {
    let handle = client.create(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));

    let elem = ElemType::Float(FloatKind::F32);

    kernel_define::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts_and_size(&handle, 2, 1, elem.size()) },
        elem,
    );

    let actual = client.read_one(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(6.0));
}

pub fn test_kernel_define_many<R: Runtime>(client: ComputeClient<R::Server>) {
    let first = client.create(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));
    let second = client.create(u32::as_bytes(&[u32::new(5), u32::new(6)]));

    let elem_first = ElemType::Float(FloatKind::F32);
    let elem_second = ElemType::UInt(UIntKind::U32);

    kernel_define_many::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts_and_size(&first, 2, 1, elem_first.size()) },
        unsafe { ArrayArg::from_raw_parts_and_size(&second, 2, 1, elem_second.size()) },
        [elem_first, elem_second],
    );

    let actual = client.read_one(first);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(7.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_numeric {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[test]
        fn test_kernel_define() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::numeric::test_kernel_define::<TestRuntime>(client);
        }

        #[test]
        fn test_kernel_define_many() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::numeric::test_kernel_define_many::<TestRuntime>(client);
        }
    };
}

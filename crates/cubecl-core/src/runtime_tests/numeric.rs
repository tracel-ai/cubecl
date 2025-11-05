use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::{ElemType, FloatKind};

#[cube(launch)]
pub fn kernel_define<N: Numeric>(array: &mut Array<N>, #[define(N)] _elem: ElemType) {
    array[UNIT_POS] += N::cast_from(5.0f32);
}

pub fn test_kernel_define<R: Runtime>(client: ComputeClient<R::Server>) {
    let handle = client.create(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));

    let vectorization = 2;
    let elem = ElemType::Float(FloatKind::F32);

    kernel_define::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts_and_size(&handle, 2, vectorization, elem.size()) },
        elem,
    );

    let actual = client.read_one(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
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
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::numeric::test_kernel_define::<TestRuntime>(client);
        }
    };
}

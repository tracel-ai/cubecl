use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_assign<F: Float>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let item = F::new(5.0);
        output[0] = item;
    }
}

pub fn test_kernel_assign_scalar<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_assign::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts(&handle, 2, vectorization) },
    );

    let actual = client.read(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_assign {
    () => {
        use super::*;

        #[test]
        fn test_assign_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_scalar::<TestRuntime, FloatT>(
                client,
            );
        }
    };
}

use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_assign<F: Float>(output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let item = F::new(5.0);
        output[0] = item;
    }
}

#[cube(launch)]
pub fn kernel_add_assign_array<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS == 0 {
        output[0] = Line::new(F::new(5.0));
        output[0] += Line::new(F::new(1.0));
    }
}

#[cube(launch)]
pub fn kernel_add_assign_line<F: Float>(output: &mut Array<Line<F>>) {
    let mut line = Line::empty(output.line_size()).fill(F::new(1.0));

    if UNIT_POS == 0 {
        #[unroll]
        for i in 0..output.line_size() {
            line[i] += F::cast_from(i);
        }
        output[0] = line;
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
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_add_assign_array<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_array::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(6.0));
}

pub fn test_kernel_add_assign_line<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_line::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, vectorization) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1.0));
    assert_eq!(actual[1], F::new(2.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_assign {
    () => {
        use super::*;

        #[test]
        fn test_assign_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_scalar::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_add_assign_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_array::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[test]
        fn test_add_assign_line() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_line::<
                TestRuntime,
                FloatType,
            >(client);
        }
    };
}

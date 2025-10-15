use crate::{self as cubecl, as_bytes, as_type};
use cubecl::prelude::*;

#[cube(launch)]
pub fn unroll_add<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let a = Line::<u32>::empty(16u32).fill(0u32);
    let b = Line::<u32>::empty(16u32).fill(3u32);

    let c = a + b;

    let mut out = Line::empty(4u32);
    #[unroll]
    for i in 0..4u32 {
        out[i] = c[i];
    }

    output[0] = Line::cast_from(out);
}

#[cube(launch)]
pub fn unroll_load_store<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let a = output[0];
    let b = Line::<F>::empty(8u32).fill(F::from_int(3));

    let c = a + b;

    output[0] = c;
}

pub fn test_unroll_add<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.empty(4 * size_of::<F>());

    unroll_add::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 4, 4) },
    );

    let actual = client.read_one(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_unroll_load_store<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes!(F: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));

    unroll_load_store::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 8, 8) },
    );

    let actual = client.read_one(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual, as_type!(F: 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unroll {
    () => {
        use super::*;

        #[test]
        fn test_unroll_add() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::unroll::test_unroll_add::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_unroll_load_store() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::unroll::test_unroll_load_store::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

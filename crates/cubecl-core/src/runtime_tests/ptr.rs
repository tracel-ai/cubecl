use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch)]
pub fn ptr_slice<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let output_ptr = Ptr::of(&output.slice_mut(0, 1));
        let slice = input.slice(2, 3);

        Ptr::as_ref_mut(&output_ptr)[0] = slice[0];
    }
}

#[cube(launch)]
pub fn ptr_primitive<F: Float>(rhs: &Array<F>, output: &mut Array<F>) {
    let value = F::cast_from(0.0);
    let value_ptr = Ptr::of(&value);
    let val = Ptr::<F>::as_ref_mut(&value_ptr);
    *val = rhs[0];

    if UNIT_POS == 0 {
        output[0] = value[0];
    }
}

#[cube(launch)]
pub fn ptr_matrix<F: Float>(rhs: &Array<F>, output: &mut Array<F>) {
    let value = F::cast_from(0.0);
    let value_ptr = Ptr::of(&value);
    let val = Ptr::<F>::as_ref_mut(&value_ptr);
    *val = rhs[0];

    if UNIT_POS == 0 {
        output[0] = value[0];
    }
}

pub fn test_ptr_base<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let input = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        ptr_slice::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

pub fn test_ptr_primitive<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let rhs = client.create(as_bytes![F: 8.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        ptr_primitive::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&rhs, 1, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(8.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_ptr {
    () => {
        use super::*;

        #[test]
        fn test_ptr_base() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::ptr::test_ptr_base::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_ptr_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::ptr::test_ptr_primitive::<TestRuntime, FloatType>(client);
        }
    };
}

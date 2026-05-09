use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_assign<F: Float>(output: &mut [F]) {
    if UNIT_POS == 0 {
        let item = F::new(5f32);
        output[0] = item;
    }
}

#[cube(launch)]
pub fn kernel_add_assign_array<F: Float, N: Size>(output: &mut [Vector<F, N>]) {
    if UNIT_POS == 0 {
        output[0] = Vector::new(F::new(5f32));
        output[0] += Vector::new(F::new(1f32));
    }
}

#[cube(launch)]
pub fn kernel_add_assign_vector<F: Float, N: Size>(output: &mut [Vector<F, N>]) {
    let mut vector = Vector::new(F::new(1f32));

    if UNIT_POS == 0 {
        #[unroll]
        for i in 0..N::value() {
            // This is awkward but shouldn't be done anyways in a real program
            vector.insert(i, vector.extract(i) + F::cast_from(i));
        }
        output[0] = vector;
    }
}

#[cube(launch)]
pub fn kernel_assign_ref<F: Float>(output: &mut [F]) {
    if UNIT_POS == 0 {
        let mut value = F::new(1f32);
        assign_ref::<F>(&mut value);
        output[0] = value;
    }
}

#[cube]
fn assign_ref<F: Float>(value: &mut F) {
    *value = F::new(5f32);
}

pub fn test_kernel_assign_scalar<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    kernel_assign::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_kernel_add_assign_array<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_array::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        vectorization,
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(6.0));
}

pub fn test_kernel_add_assign_vector<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    let vectorization = 2;

    kernel_add_assign_vector::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        vectorization,
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1.0));
    assert_eq!(actual[1], F::new(2.0));
}

pub fn test_kernel_assign_ref<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(F::as_bytes(&[F::new(0.0), F::new(1.0)]));

    kernel_assign_ref::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one(handle).unwrap();
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_assign {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_scalar::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_add_assign_array() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_array::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_add_assign_vector() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_add_assign_vector::<
                TestRuntime,
                FloatType,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_assign_ref() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::assign::test_kernel_assign_ref::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

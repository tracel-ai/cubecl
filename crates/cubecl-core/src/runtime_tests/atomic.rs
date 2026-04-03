use std::{println, vec::Vec};

use crate::{self as cubecl};

use cubecl::prelude::*;
use cubecl_ir::{StorageType, features::AtomicUsage};

#[cube(launch)]
pub fn kernel_atomic_add<I: Numeric, N: Size>(output: &mut Array<Atomic<Vector<I, N>>>) {
    if UNIT_POS == 0 {
        output[0].fetch_add(Vector::from_int(5));
    }
}

fn supports_feature<R: Runtime, F: Numeric>(
    client: &ComputeClient<R>,
    feat: AtomicUsage,
    vector_size: usize,
) -> bool {
    let ty = StorageType::Atomic(F::as_type_native_unchecked().elem_type());
    let vector = Type::new(ty).with_vector_size(vector_size);
    client.properties().atomic_type_usage(vector).contains(feat)
}

pub fn test_kernel_atomic_add<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R>,
    vector_size: usize,
) {
    if !supports_feature::<R, F>(&client, AtomicUsage::Add, vector_size) {
        println!(
            "{} Add not supported - skipped",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
        return;
    } else {
        println!(
            "{} Add supported - running",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
    }

    let data = (0..vector_size)
        .map(|_| F::from_int(12))
        .collect::<Vec<_>>();
    let handle = client.create_from_slice(F::as_bytes(&data));

    kernel_atomic_add::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&client, 1),
        vector_size,
        unsafe { ArrayArg::from_raw_parts(handle.clone(), vector_size) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert!(actual.iter().all(|actual| actual == &F::from_int(17)));
}

#[cube(launch)]
pub fn kernel_atomic_min<I: Numeric, N: Size>(output: &mut Array<Atomic<Vector<I, N>>>) {
    if UNIT_POS == 0 {
        output[0].fetch_min(Vector::from_int(5));
    }
}

pub fn test_kernel_atomic_min<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R>,
    vector_size: usize,
) {
    if !supports_feature::<R, F>(&client, AtomicUsage::MinMax, vector_size) {
        println!(
            "{} Min not supported - skipped",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
        return;
    } else {
        println!(
            "{} Min supported - running",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
    }
    let data = (0..vector_size)
        .map(|_| F::from_int(12))
        .collect::<Vec<_>>();
    let handle = client.create_from_slice(F::as_bytes(&data));

    kernel_atomic_min::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        vector_size,
        unsafe { ArrayArg::from_raw_parts(handle.clone(), vector_size) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert!(actual.iter().all(|actual| actual == &F::from_int(5)));
}

#[cube(launch)]
pub fn kernel_atomic_max<I: Numeric, N: Size>(output: &mut Array<Atomic<Vector<I, N>>>) {
    if UNIT_POS == 0 {
        output[0].fetch_max(Vector::from_int(5));
    }
}

pub fn test_kernel_atomic_max<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R>,
    vector_size: usize,
) {
    if !supports_feature::<R, F>(&client, AtomicUsage::MinMax, vector_size) {
        println!(
            "{} Max not supported - skipped",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
        return;
    } else {
        println!(
            "{} Max supported - running",
            Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size)
        );
    }
    let data = (0..vector_size)
        .map(|_| F::from_int(12))
        .collect::<Vec<_>>();
    let handle = client.create_from_slice(F::as_bytes(&data));

    kernel_atomic_max::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        vector_size,
        unsafe { ArrayArg::from_raw_parts(handle.clone(), vector_size) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert!(actual.iter().all(|actual| actual == &F::from_int(12)));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_int {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_add_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, IntType>(
                client, 1,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_min_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, IntType>(
                client, 1,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_max_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, IntType>(
                client, 1,
            );
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_float {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_add_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, FloatType>(
                client, 1,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_add_float_vec2() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, FloatType>(
                client, 2,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_add_float_vec4() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, FloatType>(
                client, 4,
            );
        }

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_min_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, FloatType>(
                client, 1,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_min_float_vec2() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, FloatType>(
                client, 2,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_min_float_vec4() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, FloatType>(
                client, 4,
            );
        }

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_max_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, FloatType>(
                client, 1,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_max_float_vec2() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, FloatType>(
                client, 2,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_atomic_max_float_vec4() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, FloatType>(
                client, 4,
            );
        }
    };
}

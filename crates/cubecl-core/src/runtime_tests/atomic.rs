use crate::{self as cubecl, AtomicFeature, Feature, ir::Elem};

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_atomic_add<I: Numeric>(output: &mut Array<Atomic<I>>) {
    if UNIT_POS == 0 {
        Atomic::add(&output[0], I::from_int(5));
    }
}

fn supports_feature<R: Runtime, F: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    feat: AtomicFeature,
) -> bool {
    match F::as_elem_native_unchecked() {
        Elem::Float(kind) => {
            client
                .properties()
                .feature_enabled(Feature::AtomicFloat(feat))
                && client
                    .properties()
                    .feature_enabled(Feature::Type(Elem::AtomicFloat(kind)))
        }
        Elem::Int(kind) => {
            client
                .properties()
                .feature_enabled(Feature::AtomicInt(feat))
                && client
                    .properties()
                    .feature_enabled(Feature::Type(Elem::AtomicInt(kind)))
        }
        Elem::UInt(kind) => {
            client
                .properties()
                .feature_enabled(Feature::AtomicUInt(feat))
                && client
                    .properties()
                    .feature_enabled(Feature::Type(Elem::AtomicUInt(kind)))
        }
        _ => unreachable!(),
    }
}

pub fn test_kernel_atomic_add<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !supports_feature::<R, F>(&client, AtomicFeature::Add) {
        println!(
            "{} Add not supported - skipped",
            Atomic::<F>::as_elem_native_unchecked()
        );
        return;
    };

    let handle = client.create(F::as_bytes(&[F::from_int(12), F::from_int(1)]));

    kernel_atomic_add::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::from_int(17));
}

#[cube(launch)]
pub fn kernel_atomic_min<I: Numeric>(output: &mut Array<Atomic<I>>) {
    if UNIT_POS == 0 {
        Atomic::min(&output[0], I::from_int(5));
    }
}

pub fn test_kernel_atomic_min<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !supports_feature::<R, F>(&client, AtomicFeature::MinMax) {
        println!(
            "{} Min not supported - skipped",
            Atomic::<F>::as_elem_native_unchecked()
        );
        return;
    };

    let handle = client.create(F::as_bytes(&[F::from_int(12), F::from_int(1)]));

    kernel_atomic_min::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::from_int(5));
}

#[cube(launch)]
pub fn kernel_atomic_max<I: Numeric>(output: &mut Array<Atomic<I>>) {
    if UNIT_POS == 0 {
        Atomic::max(&output[0], I::from_int(5));
    }
}

pub fn test_kernel_atomic_max<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !supports_feature::<R, F>(&client, AtomicFeature::MinMax) {
        println!(
            "{} Max not supported - skipped",
            Atomic::<F>::as_elem_native_unchecked()
        );
        return;
    };

    let handle = client.create(F::as_bytes(&[F::from_int(12), F::from_int(1)]));

    kernel_atomic_max::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::from_int(12));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_int {
    () => {
        use super::*;

        #[test]
        fn test_atomic_add_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, IntType>(
                client,
            );
        }

        #[test]
        fn test_atomic_min_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, IntType>(
                client,
            );
        }

        #[test]
        fn test_atomic_max_int() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, IntType>(
                client,
            );
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_float {
    () => {
        use super::*;

        #[test]
        fn test_atomic_add_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_add::<TestRuntime, FloatType>(
                client,
            );
        }

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        #[test]
        fn test_atomic_min_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_min::<TestRuntime, FloatType>(
                client,
            );
        }

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        #[test]
        fn test_atomic_max_float() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::atomic::test_kernel_atomic_max::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

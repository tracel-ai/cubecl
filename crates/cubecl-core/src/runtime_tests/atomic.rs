use crate::{self as cubecl, ir::Elem, AtomicFeature, Feature};

use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_atomic_add<I: Numeric>(output: &mut Array<Atomic<I>>) {
    if UNIT_POS == 0 {
        Atomic::add(&output[0], I::from_int(5));
    }
}

fn supports_feature<R: Runtime, F: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    float_feat: AtomicFeature,
) -> bool {
    match F::as_elem_native_unchecked() {
        Elem::Float(kind) => {
            client
                .properties()
                .feature_enabled(Feature::FloatAtomic(float_feat))
                && client
                    .properties()
                    .feature_enabled(Feature::Type(Elem::AtomicFloat(kind)))
        }
        Elem::Int(kind) => client
            .properties()
            .feature_enabled(Feature::Type(Elem::AtomicInt(kind))),
        Elem::UInt(kind) => client
            .properties()
            .feature_enabled(Feature::Type(Elem::AtomicUInt(kind))),
        _ => unreachable!(),
    }
}

pub fn test_kernel_atomic_add<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !supports_feature::<R, F>(&client, AtomicFeature::Add) {
        println!("Not supported - skipped");
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
    };
}

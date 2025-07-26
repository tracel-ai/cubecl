use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch)]
pub fn sequence_for_loop<F: Float>(output: &mut Array<F>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut sequence = Sequence::<F>::new();
    sequence.push(F::new(1.0));
    sequence.push(F::new(4.0));

    for value in sequence {
        output[0] += value;
    }
}

#[cube(launch)]
pub fn sequence_index<F: Float>(output: &mut Array<F>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut sequence = Sequence::<F>::new();
    sequence.push(F::new(2.0));
    sequence.push(F::new(4.0));

    output[0] += *sequence.index(0);
    output[0] += *Sequence::<F>::index(&sequence, 1);
}

pub fn test_sequence_for_loop<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0]).expect("Alloc failed");

    sequence_for_loop::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_sequence_index<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    let handle = client.create(as_bytes![F: 0.0]).expect("Alloc failed");

    sequence_index::launch::<F, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        unsafe { ArrayArg::from_raw_parts::<F>(&handle, 2, 1) },
    );

    let actual = client.read_one(handle.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(6.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_sequence {
    () => {
        use super::*;

        #[test]
        fn test_sequence_for_loop() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::sequence::test_sequence_for_loop::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_sequence_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::sequence::test_sequence_index::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}

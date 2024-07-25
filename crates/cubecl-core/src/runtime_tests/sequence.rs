use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch)]
pub fn sequence_for_loop(output: &mut Array<F32>) {
    if UNIT_POS != UInt::new(0) {
        return;
    }

    let mut sequence = Sequence::<F32>::new();
    sequence.push(F32::new(1.0));
    sequence.push(F32::new(4.0));

    for value in sequence {
        output[0] += value;
    }
}

#[cube(launch)]
pub fn sequence_index(output: &mut Array<F32>) {
    if UNIT_POS != UInt::new(0) {
        return;
    }

    let mut sequence = Sequence::<F32>::new();
    sequence.push(F32::new(2.0));
    sequence.push(F32::new(4.0));

    output[0] += *sequence.index(0);
    output[0] += *Sequence::index(&sequence, 1);
}

pub fn test_sequence_for_loop<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0]));

    sequence_for_loop::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ArrayArg::new(&handle, 2),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 5.0);
}

pub fn test_sequence_index<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle = client.create(f32::as_bytes(&[0.0]));

    sequence_index::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::default(),
        ArrayArg::new(&handle, 2),
    );

    let actual = client.read(handle.binding());
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 6.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_sequence {
    () => {
        use super::*;

        #[test]
        fn test_sequence_for_loop() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::sequence::test_sequence_for_loop::<TestRuntime>(client);
        }

        #[test]
        fn test_sequence_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::sequence::test_sequence_index::<TestRuntime>(client);
        }
    };
}

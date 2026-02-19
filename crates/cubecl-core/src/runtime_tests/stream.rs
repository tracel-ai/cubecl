use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::stream_id::StreamId;

#[cube(launch)]
pub fn big_task<F: Float>(input: &Array<u32>, output: &mut Array<F>, num_loop: usize) {
    if ABSOLUTE_POS > output.len() {
        terminate!()
    }

    for i in 0..num_loop {
        let pos = i % input.len();
        output[ABSOLUTE_POS] += F::cast_from(input[pos]) / F::cast_from(num_loop);
    }
}

pub fn test_stream<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let client_1 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10000 });
        c
    };
    let client_2 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10001 });
        c
    };

    let len = 4096;
    let input: Vec<u32> = (0..len as u32).collect();
    let mut input = client_1.create_from_slice(u32::as_bytes(&input));
    let mut output = None;

    for _ in 0..300 {
        let output_ = client_1.empty(len * core::mem::size_of::<F>());
        unsafe {
            big_task::launch::<F, R>(
                &client_1,
                CubeCount::Static(len as u32 / 32, 1, 1),
                CubeDim::new_1d(32),
                ArrayArg::from_raw_parts::<F>(&input, len, 1),
                ArrayArg::from_raw_parts::<F>(&output_, len, 1),
                ScalarArg::new(4096),
            )
        };
        input = output_.clone();
        output = Some(output_);
    }

    let actual = client_2.read_one_unchecked(output.unwrap());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1318936000.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_stream {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        #[ignore = "Not yet supported by all backends"]
        fn test_stream() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::stream::test_stream::<TestRuntime, FloatType>(client);
        }
    };
}

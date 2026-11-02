use crate::{self as cubecl};
use alloc::{vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
pub fn big_task<F: Float>(input: &[u32], output: &mut [F], num_loop: usize) {
    if ABSOLUTE_POS > output.len() {
        terminate!()
    }

    for i in 0..num_loop {
        let pos = i % input.len();
        output[ABSOLUTE_POS] += F::cast_from(input[pos]) / F::cast_from(num_loop);
    }
}

pub fn test_stream<R: Runtime, F: Float + CubeElement>(client: Client) {
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
            big_task::launch::<F>(
                &client_1,
                CubeCount::Static(len as u32 / 32, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input, len),
                BufferArg::from_raw_parts(output_.clone(), len),
                4096,
            )
        };
        input = output_.clone();
        output = Some(output_);
    }

    let actual = client_2.read_one_unchecked(output.unwrap());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1318936000.0));
}

#[cube(launch)]
pub fn add_one(buffer: &mut [f32]) {
    if ABSOLUTE_POS < buffer.len() {
        buffer[ABSOLUTE_POS] += 1.0;
    }
}

#[cube(launch)]
pub fn copy(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// A buffer created on one stream, written in place on a second and read on a third, which shares
/// no other buffer with the writer.
pub fn test_write_on_another_stream_is_read(client: Client) {
    let on_stream = |value| unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value });
        c
    };
    let (creator, writer, reader) = (on_stream(20000), on_stream(20001), on_stream(20002));

    let len = 256;
    let buffer = creator.create_from_slice(f32::as_bytes(&vec![0.0; len]));
    unsafe {
        add_one::launch(
            &writer,
            CubeCount::Static(len as u32 / 32, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(buffer.clone(), len),
        )
    };
    let output = reader.empty(len * core::mem::size_of::<f32>());
    unsafe {
        copy::launch(
            &reader,
            CubeCount::Static(len as u32 / 32, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(buffer, len),
            BufferArg::from_raw_parts(output.clone(), len),
        )
    };

    let actual = reader.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), vec![1.0; len]);
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

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_stream_untyped {
    () => {
        mod stream {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_write_on_another_stream_is_read() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::stream::test_write_on_another_stream_is_read(client);
            }
        }
    };
}

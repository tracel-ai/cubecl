//! Cross-stream correctness tests for the native Metal backend.
//!
//! `StreamId` is derived from the OS thread, so a single [`ComputeClient`] drives a
//! different stream per thread, and each stream owns its own `memory_management`. A
//! buffer therefore lives only in its origin stream's manager. These tests allocate a
//! binding on one stream and consume it on another to exercise cross-stream resolution
//! and synchronization.

use cubecl_core::{self as cubecl, prelude::*};

type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn add_one_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1;
    }
}

#[cube(launch_unchecked)]
fn copy_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

fn client() -> ComputeClient<R> {
    let device = Default::default();
    R::client(&device)
}

/// Allocate an input on a spawned thread (a different `StreamId`), then consume it on
/// the main thread, where it must be resolved from the spawned stream's memory manager.
#[test]
fn cross_thread_binding_read_on_other_stream() {
    let client = client();

    // Produce the input on another thread => binding.stream != main stream.
    let input = {
        let client = client.clone();
        std::thread::spawn(move || client.create_from_slice(u32::as_bytes(&[10, 20, 30, 40])))
            .join()
            .unwrap()
    };

    let n = 4usize;
    let output = client.empty(n * core::mem::size_of::<u32>());

    unsafe {
        add_one_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(input.clone(), n),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(output.clone());
    let actual = u32::from_bytes(&bytes);

    assert_eq!(actual, &[11, 21, 31, 41]);
}

/// Iterated to widen the race window for a missed cross-stream wait: the producer thread
/// writes the data and the consumer must observe those writes.
#[test]
fn cross_thread_producer_consumer_dependency() {
    let client = client();
    let n = 256usize;

    for iter in 0..100u32 {
        let base = iter;
        let expected: Vec<u32> = (0..n as u32).map(|i| i.wrapping_add(base)).collect();

        // Producer thread: allocate + fill the input on its own stream.
        let input = {
            let client = client.clone();
            let data = expected.clone();
            std::thread::spawn(move || client.create_from_slice(u32::as_bytes(&data)))
                .join()
                .unwrap()
        };

        let output = client.empty(n * core::mem::size_of::<u32>());

        unsafe {
            copy_kernel::launch_unchecked::<R>(
                &client,
                CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(input.clone(), n),
                BufferArg::from_raw_parts(output.clone(), n),
            );
        }

        let bytes = client.read_one_unchecked(output.clone());
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, expected.as_slice(), "mismatch on iteration {iter}");
    }
}

/// Cross-stream dynamic dispatch: the indirect cube-count buffer is produced on a
/// different stream than the launch. If its origin stream isn't synced before the GPU
/// reads it, the dispatch dims are stale.
#[test]
fn cross_thread_dynamic_cube_count() {
    let client = client();
    let n = 128usize;
    let groups = n.div_ceil(64) as u32;

    for iter in 0..50u32 {
        let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_add(iter)).collect();

        // Produce BOTH the input and the indirect dispatch buffer on another stream.
        let (input, count) = {
            let client = client.clone();
            let data = data.clone();
            std::thread::spawn(move || {
                let input = client.create_from_slice(u32::as_bytes(&data));
                let count = client.create_from_slice(u32::as_bytes(&[groups, 1, 1]));
                (input, count)
            })
            .join()
            .unwrap()
        };

        let output = client.empty(n * core::mem::size_of::<u32>());

        unsafe {
            copy_kernel::launch_unchecked::<R>(
                &client,
                CubeCount::Dynamic(count.binding()),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(input.clone(), n),
                BufferArg::from_raw_parts(output.clone(), n),
            );
        }

        let bytes = client.read_one_unchecked(output.clone());
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, data.as_slice(), "mismatch on iteration {iter}");
    }
}

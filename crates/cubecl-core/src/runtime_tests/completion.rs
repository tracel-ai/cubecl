use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use cubecl_environment::stream::StreamId;
use cubecl_ir::features::AtomicUsage;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn write_value(output: &mut [u32], value: u32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = value;
    }
}

/// Hold the stream until another stream publishes the release flag. The flag is
/// an atomic so its read is not hoisted out of the loop.
#[cube(launch)]
fn wait_for_flag(output: &mut [u32], flag: &[Atomic<u32>]) {
    let mut released = flag[0].load();
    while released == 0 {
        released = flag[0].load();
    }
    output[ABSOLUTE_POS] = released;
}

#[cube(launch)]
fn release_flag(flag: &[Atomic<u32>]) {
    flag[0].store(1);
}

/// A launch before the boundary is complete once the handle is waited on, and
/// the result is readable then.
pub fn test_completion_after_launch<R: Runtime>(client: Client) {
    let output = client.empty(size_of::<u32>());
    unsafe {
        write_value::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 1),
            42,
        )
    };

    let completion = client.completion();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !completion.is_complete().unwrap() {
        assert!(
            std::time::Instant::now() < deadline,
            "the completion never reported itself complete"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    completion.wait().unwrap();
    assert!(completion.is_complete().unwrap());

    let bytes = client.read_one_unchecked(output);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[42]);

    // Waiting more than once is valid.
    completion.wait().unwrap();
}

/// The boundary is fixed at the call: a launch submitted afterwards is not part
/// of it, and the second handle still sees its own work.
///
/// The later work is gated on a flag another stream releases, so this observes
/// the first boundary pass while the second is still held. It needs two streams:
/// wgpu puts both on one queue, where a later write cannot pass the gate, so
/// that backend does not generate this test.
pub fn test_completion_boundary_is_fixed<R: Runtime>(client: Client) {
    let ty = Type::atomic(u32::elem_type_native());
    if !client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::LoadStore)
    {
        return;
    }
    // The gate blocks one worker while the release runs on another.
    if client.properties().hardware.max_units_per_cube < 2 {
        return;
    }

    // The flag lives on its own stream, so releasing it does not wait on the
    // gated stream's work.
    let mut release_client = client.clone();
    unsafe {
        release_client.set_stream(StreamId { value: 10_000 });
    }
    let flag = release_client.create_from_slice(u32::as_bytes(&[0u32]));
    let output = client.empty(size_of::<u32>());

    let first = client.completion();
    unsafe {
        wait_for_flag::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 1),
            BufferArg::from_raw_parts(flag.clone(), 1),
        )
    };
    let second = client.completion();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let first_completed = loop {
        match first.is_complete() {
            Ok(true) => break true,
            Ok(false) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Ok(false) => break false,
            // A backend that cannot query the handle is not this test's concern;
            // leave the gate before reporting.
            Err(_) => break false,
        }
    };
    let second_incomplete = !second.is_complete().unwrap();

    // Release the gate even when the assertion below is about to fail, so a
    // broken boundary cannot leave a kernel spinning for the rest of the run.
    // A CPU stream publishes from another worker; a device stream uses a host
    // write, whose copy engine can run while the gate kernel occupies the
    // compute queue.
    if client.name() == "cpu" {
        unsafe {
            release_flag::launch(
                &release_client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(flag.clone(), 1),
            )
        };
    } else {
        release_client.write(&flag, Bytes::from_elems([1u32].to_vec()));
    }
    release_client.flush().unwrap();
    second.wait().unwrap();

    assert!(
        first_completed,
        "the first completion waited for work submitted after it"
    );
    assert!(
        second_incomplete,
        "the second completion passed the gate before it was released"
    );

    let bytes = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&bytes), &[1]);
}

/// The same handle can be waited on from several threads at once.
pub fn test_completion_concurrent_wait<R: Runtime>(client: Client) {
    let output = client.empty(size_of::<u32>());
    unsafe {
        write_value::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(output.clone(), 1),
            7,
        )
    };

    let completion = client.completion();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let completion = completion.clone();
            scope.spawn(move || {
                completion.wait().unwrap();
                assert!(completion.is_complete().unwrap());
            });
        }
    });

    completion.wait().unwrap();
    let bytes = client.read_one_unchecked(output);
    assert_eq!(
        u32::from_bytes(&bytes),
        &[7],
        "every waiter names the same boundary"
    );
}

/// A clone names the same boundary, and the last one to drop releases the
/// backend's resources.
pub fn test_completion_clone_and_drop<R: Runtime>(client: Client) {
    let completion = client.completion();
    let clone = completion.clone();
    drop(completion);
    clone.wait().unwrap();
    assert!(clone.is_complete().unwrap());
    drop(clone);
}

/// A backend with no completion support reports it from the handle.
pub fn test_completion_unsupported<R: Runtime>(client: Client) {
    let completion = client.completion();
    assert!(
        completion.wait().is_err(),
        "an unsupported backend has to report the error, not hang"
    );
    assert!(completion.is_complete().is_err());
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_completion {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_completion_after_launch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::completion::test_completion_after_launch::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_completion_concurrent_wait() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::completion::test_completion_concurrent_wait::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_completion_clone_and_drop() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::completion::test_completion_clone_and_drop::<TestRuntime>(
                client,
            );
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_completion_boundary {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_completion_boundary_is_fixed() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::completion::test_completion_boundary_is_fixed::<
                TestRuntime,
            >(client);
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_completion_unsupported {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_completion_unsupported() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::completion::test_completion_unsupported::<TestRuntime>(
                client,
            );
        }
    };
}

//! Kernels that read and write the memory of another device of their runtime.

use crate as cubecl;
use crate::prelude::*;
use alloc::{string::ToString, vec, vec::Vec};
use cubecl_common::device::Device;
use cubecl_environment::future::block_on;
use cubecl_runtime::{client::Client, runtime::Runtime, server::Handle};

const LEN: usize = 64;

#[cube(launch)]
fn scale_from_peer(input: &Peer<[f32]>, output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] * 2.0;
    }
}

#[cube(launch)]
fn scale_to_peer(input: &[f32], output: &mut Peer<[f32]>) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] * 2.0;
    }
}

#[cube(launch)]
fn scale_into(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] * 2.0;
    }
}

/// A launch the compiler refuses, writing a peer buffer it never reaches.
#[cube(launch_unchecked)]
fn rejected_into_peer(out: &mut Peer<[u32]>, #[comptime] reason: alloc::string::String) {
    push_validation_error(reason);
    out[0] = 1u32;
}

#[cube(launch)]
fn copy_from_peer(input: &Peer<[u32]>, out: &mut [u32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// Two clients of the runtime whose kernels can bind each other's memory, or
/// `None` when the machine has no such pair.
fn peers<R: Runtime>() -> Option<(Client, Client)> {
    let devices = R::enumerate_devices(0);
    for (i, a) in devices.iter().enumerate() {
        for b in &devices[i + 1..] {
            let a = R::client(&R::Device::from_id(*a));
            let b = R::client(&R::Device::from_id(*b));
            if a.can_access_peer(&b) && b.can_access_peer(&a) {
                return Some((a, b));
            }
        }
    }
    std::println!("skipping peer access: no two devices can access each other's memory");
    None
}

fn values(len: usize) -> Vec<f32> {
    (0..len).map(|i| i as f32 + 1.0).collect()
}

fn zeros(client: &Client, len: usize) -> Handle {
    client.create_from_slice(u32::as_bytes(&vec![0u32; len]))
}

fn peer_arg(handle: &Handle, len: usize) -> PeerArg {
    unsafe { PeerArg::from_raw_parts(handle, len) }
}

fn scale_from(client: &Client, input: &Handle, output: Handle, len: usize) {
    scale_from_peer::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(len as u32),
        peer_arg(input, len),
        unsafe { BufferArg::from_raw_parts(output, len) },
    );
}

fn scale_to(client: &Client, input: Handle, output: &Handle, len: usize) {
    scale_to_peer::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(len as u32),
        unsafe { BufferArg::from_raw_parts(input, len) },
        peer_arg(output, len),
    );
}

fn scale(client: &Client, input: Handle, output: Handle, len: usize) {
    scale_into::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(len as u32),
        unsafe { BufferArg::from_raw_parts(input, len) },
        unsafe { BufferArg::from_raw_parts(output, len) },
    );
}

pub fn test_a_device_is_not_its_own_peer<R: Runtime>() {
    let client = R::client(&Default::default());
    assert!(!client.can_access_peer(&client));
}

pub fn test_a_kernel_reads_a_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
    let input = remote.share_with_peers(&input).unwrap();
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    // Nothing orders the peer's upload before the kernel reading it.
    block_on(remote.sync()).unwrap();

    scale_from(&local, &input, output.clone(), LEN);

    let actual = local.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN).iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
}

pub fn test_a_kernel_writes_a_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = local.create_from_slice(f32::as_bytes(&values(LEN)));
    let output = remote.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    let output = remote.share_with_peers(&output).unwrap();
    block_on(remote.sync()).unwrap();

    scale_to(&local, input, &output, LEN);
    // Nothing orders the kernel's writes before the owner's read.
    block_on(local.sync()).unwrap();

    let actual = remote.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN).iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
}

/// The owner's own kernels take a shared buffer through a `Peer` parameter
/// too: on its device it is an ordinary buffer.
pub fn test_a_peer_parameter_binds_memory_of_its_own_device<R: Runtime>() {
    let Some((local, _remote)) = peers::<R>() else {
        return;
    };
    let input = local.create_from_slice(f32::as_bytes(&values(LEN)));
    let input = local.share_with_peers(&input).unwrap();
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));

    scale_from(&local, &input, output.clone(), LEN);

    let actual = local.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN).iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
}

pub fn test_a_kernel_reads_a_slice_of_a_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    const SKIP: usize = 16;
    let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
    let input = remote.share_with_peers(&input).unwrap();
    let slice = input.offset_start((SKIP * size_of::<f32>()) as u64);
    let output = local.create_from_slice(f32::as_bytes(&[0.0; LEN - SKIP]));
    block_on(remote.sync()).unwrap();

    scale_from(&local, &slice, output.clone(), LEN - SKIP);

    let actual = local.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN)[SKIP..].iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
}

/// The kernel holds the peer's memory: the owner dropping every handle to it
/// right after the launch leaves it allocated until the kernel completed.
pub fn test_a_peer_buffer_outlives_its_handles_while_bound<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    {
        let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
        let input = remote.share_with_peers(&input).unwrap();
        block_on(remote.sync()).unwrap();
        scale_from(&local, &input, output.clone(), LEN);
    }
    // The owner reuses freed memory first: a buffer of the same size filled
    // with something else lands where the input was, had it been freed.
    let other = remote.create_from_slice(f32::as_bytes(&vec![-1.0; LEN]));
    block_on(remote.sync()).unwrap();

    let actual = local.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN).iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
    drop(other);
}

pub fn test_an_unshared_foreign_handle_is_refused<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    let launch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scale_from(&local, &input, output, LEN)
    }));
    assert!(launch.is_err(), "a handle never shared must not bind");
}

/// Only a `Peer` parameter binds another device's memory: the kernel would
/// otherwise make promises about it that another device breaks.
pub fn test_a_shared_handle_in_a_plain_parameter_is_refused<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
    let input = remote.share_with_peers(&input).unwrap();
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    let launch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scale(&local, input, output, LEN)
    }));
    assert!(
        launch.is_err(),
        "a buffer of another device must not bind through a plain slice"
    );
}

/// Elements the handshake kernels exchange: several cubes' worth, so every cube of the producer
/// has to arrive before the consumer reads.
const HANDOFF_LEN: usize = 1024;
const HANDOFF_UNITS: u32 = 256;
const HANDOFF_CUBES: u32 = (HANDOFF_LEN as u32) / HANDOFF_UNITS;
/// How long a consumer waits for the flag before giving up, so a missing signal fails the test
/// rather than hanging the device.
const SPIN_LIMIT: u32 = 20_000_000;

/// Write `token` everywhere in `data`, publish it at system scope, then raise `flag` once per
/// cube. The atomic on a `Peer` buffer synchronizes at system scope without being asked.
#[cube(launch)]
fn produce(data: &mut Peer<[u32]>, flag: &mut Peer<[Atomic<u32>]>, token: u32) {
    let i = ABSOLUTE_POS;
    if i < data.len() {
        data[i] = token;
    }
    // Release: what every unit of the cube wrote is visible to whoever acquires after the flag.
    sync_system();
    if UNIT_POS == 0 {
        flag[0].fetch_add(1u32);
    }
}

/// Wait until `flag` reaches `arrivals`, then copy `data` to `out`. Each cube records the flag it
/// saw in `seen`, so a consumer that gave up waiting tells apart from one that read stale data.
#[cube(launch)]
fn consume(
    data: &Peer<[u32]>,
    flag: &mut Peer<[Atomic<u32>]>,
    out: &mut [u32],
    seen_by_cube: &mut [u32],
    arrivals: u32,
    spin_limit: u32,
) {
    let mut seen = Shared::<u32>::new();
    if UNIT_POS == 0 {
        let mut spins = 0u32;
        let mut arrived = flag[0].load();
        while arrived < arrivals && spins < spin_limit {
            spins += 1u32;
            arrived = flag[0].load();
        }
        *seen = arrived;
        seen_by_cube[CUBE_POS] = arrived;
    }
    // Acquire: what the producer published before raising the flag is visible here.
    sync_system();
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = data[i];
    }
}

/// Launch the consumer, returning the buffers it fills: the data it read, and the flag each
/// cube saw.
fn launch_consumer(client: &Client, data: &Handle, flag: &Handle) -> (Handle, Handle) {
    let out = zeros(client, HANDOFF_LEN);
    let seen = zeros(client, HANDOFF_CUBES as usize);
    consume::launch(
        client,
        CubeCount::Static(HANDOFF_CUBES, 1, 1),
        CubeDim::new_1d(HANDOFF_UNITS),
        peer_arg(data, HANDOFF_LEN),
        peer_arg(flag, 1),
        unsafe { BufferArg::from_raw_parts(out.clone(), HANDOFF_LEN) },
        unsafe { BufferArg::from_raw_parts(seen.clone(), HANDOFF_CUBES as usize) },
        HANDOFF_CUBES,
        SPIN_LIMIT,
    );
    (out, seen)
}

/// What the consumer read is the producer's `token`, after every cube saw every producer
/// cube arrive.
fn assert_handed_off(consumer: &Client, out: Handle, seen: Handle, token: u32) {
    let seen = consumer.read_one(seen).unwrap();
    assert_eq!(
        u32::from_bytes(&seen),
        vec![HANDOFF_CUBES; HANDOFF_CUBES as usize],
        "token {token}: the flag each consumer cube saw"
    );
    let actual = consumer.read_one(out).unwrap();
    let actual = u32::from_bytes(&actual);
    let stale = actual.iter().filter(|value| **value != token).count();
    assert_eq!(
        stale,
        0,
        "token {token}: {stale} values read stale, first {:?}",
        &actual[..8]
    );
}

fn launch_producer(client: &Client, data: &Handle, flag: &Handle, token: u32) {
    produce::launch(
        client,
        CubeCount::Static(HANDOFF_CUBES, 1, 1),
        CubeDim::new_1d(HANDOFF_UNITS),
        peer_arg(data, HANDOFF_LEN),
        peer_arg(flag, 1),
        token,
    );
}

/// The producer writes into the consumer's memory and raises a flag there, while the consumer
/// is already running and waiting on it: the two kernels hand data off with no host in between.
pub fn test_a_kernel_pushes_data_to_a_waiting_peer_kernel<R: Runtime>() {
    let Some((producer, consumer)) = peers::<R>() else {
        return;
    };
    for token in [7u32, 11, 13] {
        let data = consumer
            .share_with_peers(&zeros(&consumer, HANDOFF_LEN))
            .unwrap();
        let flag = consumer.share_with_peers(&zeros(&consumer, 1)).unwrap();
        block_on(consumer.sync()).unwrap();

        let (out, seen) = launch_consumer(&consumer, &data, &flag);
        launch_producer(&producer, &data, &flag, token);

        assert_handed_off(&consumer, out, seen, token);
        block_on(producer.sync()).unwrap();
    }
}

/// The producer writes its own memory and raises a flag in the consumer's, and the consumer
/// reads the data from the producer's memory once it saw the flag.
pub fn test_a_kernel_pulls_data_a_peer_kernel_published<R: Runtime>() {
    let Some((producer, consumer)) = peers::<R>() else {
        return;
    };
    for token in [7u32, 11, 13] {
        let data = producer
            .share_with_peers(&zeros(&producer, HANDOFF_LEN))
            .unwrap();
        let flag = consumer.share_with_peers(&zeros(&consumer, 1)).unwrap();
        block_on(producer.sync()).unwrap();
        block_on(consumer.sync()).unwrap();

        let (out, seen) = launch_consumer(&consumer, &data, &flag);
        launch_producer(&producer, &data, &flag, token);

        assert_handed_off(&consumer, out, seen, token);
        block_on(producer.sync()).unwrap();
    }
}

fn launch_rejected_into_peer(client: &Client, out: &Handle, reason: &str) {
    unsafe {
        rejected_into_peer::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            peer_arg(out, 1),
            reason.into(),
        )
    };
}

fn launch_copy_from_peer(client: &Client, input: &Handle, out: Handle, len: usize) {
    copy_from_peer::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_1d(len as u32),
        peer_arg(input, len),
        unsafe { BufferArg::from_raw_parts(out, len) },
    );
}

/// Shared memory carried over to a handle of other memory does not bind: the
/// address would point a kernel at bytes the handle does not name.
pub fn test_shared_memory_of_another_buffer_is_refused<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let shared = remote.share_with_peers(&zeros(&remote, 1)).unwrap();
    let mut forged = zeros(&remote, 1);
    forged.peer = shared.peer.clone();
    let out = zeros(&local, 1);
    let launch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_copy_from_peer(&local, &forged, out, 1)
    }));
    assert!(
        launch.is_err(),
        "a handle carrying another buffer's address must not bind"
    );
}

/// A peer kernel that never ran leaves the memory it was going to write on its
/// owner carrying the failure, so a read there names the cause.
pub fn test_a_failed_peer_kernel_taints_the_memory_it_was_writing<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let output = remote.share_with_peers(&zeros(&remote, 1)).unwrap();
    block_on(remote.sync()).unwrap();

    launch_rejected_into_peer(&local, &output, "peer writer");
    block_on(local.sync()).unwrap();

    let err = remote
        .read_one(output)
        .expect_err("the peer kernel never wrote the buffer")
        .to_string();
    assert!(
        err.contains("peer writer"),
        "the read must name the peer's failure, got: {err}"
    );
}

/// A peer kernel reading memory whose owner failed to write it does not run,
/// and its outputs carry the owner's failure. Writing the memory again on its
/// owner lets the peer read it.
pub fn test_a_peer_kernel_does_not_run_on_a_failed_owner_write<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = remote.share_with_peers(&zeros(&remote, 1)).unwrap();
    unsafe {
        super::stream_errors::rejected::launch_unchecked(
            &remote,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input.clone(), 1),
            "owner writer".into(),
        )
    };
    block_on(remote.sync()).unwrap();

    let out = zeros(&local, 1);
    launch_copy_from_peer(&local, &input, out.clone(), 1);
    let err = local
        .read_one(out)
        .expect_err("the peer kernel must not run on a failed input")
        .to_string();
    assert!(
        err.contains("owner writer"),
        "the read must name the owner's failure, got: {err}"
    );

    remote.write(
        &input,
        cubecl_common::bytes::Bytes::from_bytes_vec(u32::as_bytes(&[5u32]).to_vec()),
    );
    block_on(remote.sync()).unwrap();
    let out = zeros(&local, 1);
    launch_copy_from_peer(&local, &input, out.clone(), 1);
    assert_eq!(u32::from_bytes(&local.read_one(out).unwrap()), [5]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_peer {
    () => {
        mod peer {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_a_device_is_not_its_own_peer() {
                cubecl_core::runtime_tests::peer::test_a_device_is_not_its_own_peer::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_kernel_reads_a_peer_buffer() {
                cubecl_core::runtime_tests::peer::test_a_kernel_reads_a_peer_buffer::<TestRuntime>();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_kernel_writes_a_peer_buffer() {
                cubecl_core::runtime_tests::peer::test_a_kernel_writes_a_peer_buffer::<TestRuntime>(
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_kernel_reads_a_slice_of_a_peer_buffer() {
                cubecl_core::runtime_tests::peer::test_a_kernel_reads_a_slice_of_a_peer_buffer::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_peer_buffer_outlives_its_handles_while_bound() {
                cubecl_core::runtime_tests::peer::test_a_peer_buffer_outlives_its_handles_while_bound::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_kernel_pushes_data_to_a_waiting_peer_kernel() {
                cubecl_core::runtime_tests::peer::test_a_kernel_pushes_data_to_a_waiting_peer_kernel::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_kernel_pulls_data_a_peer_kernel_published() {
                cubecl_core::runtime_tests::peer::test_a_kernel_pulls_data_a_peer_kernel_published::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_shared_memory_of_another_buffer_is_refused() {
                cubecl_core::runtime_tests::peer::test_shared_memory_of_another_buffer_is_refused::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_failed_peer_kernel_taints_the_memory_it_was_writing() {
                cubecl_core::runtime_tests::peer::test_a_failed_peer_kernel_taints_the_memory_it_was_writing::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_peer_kernel_does_not_run_on_a_failed_owner_write() {
                cubecl_core::runtime_tests::peer::test_a_peer_kernel_does_not_run_on_a_failed_owner_write::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_peer_parameter_binds_memory_of_its_own_device() {
                cubecl_core::runtime_tests::peer::test_a_peer_parameter_binds_memory_of_its_own_device::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_a_shared_handle_in_a_plain_parameter_is_refused() {
                cubecl_core::runtime_tests::peer::test_a_shared_handle_in_a_plain_parameter_is_refused::<
                    TestRuntime,
                >();
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_an_unshared_foreign_handle_is_refused() {
                cubecl_core::runtime_tests::peer::test_an_unshared_foreign_handle_is_refused::<
                    TestRuntime,
                >();
            }
        }
    };
}

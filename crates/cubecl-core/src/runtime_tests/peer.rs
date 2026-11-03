//! Kernels that read and write the memory of another device of their runtime.

use crate as cubecl;
use crate::prelude::*;
use alloc::{vec, vec::Vec};
use cubecl_common::device::Device;
use cubecl_environment::future::block_on;
use cubecl_runtime::{client::Client, runtime::Runtime, server::Handle};

const LEN: usize = 64;

#[cube(launch)]
fn scale_into(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] * 2.0;
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
    let input = remote.share_with_peers(input).unwrap();
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN]));
    // Nothing orders the peer's upload before the kernel reading it.
    block_on(remote.sync()).unwrap();

    scale(&local, input, output.clone(), LEN);

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
    let output = remote.share_with_peers(output).unwrap();
    block_on(remote.sync()).unwrap();

    scale(&local, input, output.clone(), LEN);
    // Nothing orders the kernel's writes before the owner's read.
    block_on(local.sync()).unwrap();

    let actual = remote.read_one(output).unwrap();
    let expected: Vec<f32> = values(LEN).iter().map(|v| v * 2.0).collect();
    assert_eq!(f32::from_bytes(&actual), expected);
}

pub fn test_a_kernel_reads_a_slice_of_a_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    const SKIP: usize = 16;
    let input = remote.create_from_slice(f32::as_bytes(&values(LEN)));
    let input = remote.share_with_peers(input).unwrap();
    let slice = input.offset_start((SKIP * size_of::<f32>()) as u64);
    let output = local.create_from_slice(f32::as_bytes(&vec![0.0; LEN - SKIP]));
    block_on(remote.sync()).unwrap();

    scale(&local, slice, output.clone(), LEN - SKIP);

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
        let input = remote.share_with_peers(input).unwrap();
        block_on(remote.sync()).unwrap();
        scale(&local, input, output.clone(), LEN);
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
        scale(&local, input, output, LEN)
    }));
    assert!(launch.is_err(), "a handle never shared must not bind");
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
            fn test_an_unshared_foreign_handle_is_refused() {
                cubecl_core::runtime_tests::peer::test_an_unshared_foreign_handle_is_refused::<
                    TestRuntime,
                >();
            }
        }
    };
}

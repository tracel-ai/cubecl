use alloc::{vec, vec::Vec};
use cubecl_common::device::Device;
use cubecl_core::prelude::*;
use cubecl_environment::future::block_on;
use cubecl_runtime::{client::Client, runtime::Runtime, server::PeerAccess};

use crate::throughput::link_peer::{PeerCopier, VECTOR_BYTES, launch_peer_load, launch_peer_store};

/// Past one launch's grid, so units stride, and not a power of two.
const VECTORS: u64 = 1024 * 256 + 1000;

const BYTES: u64 = VECTORS * VECTOR_BYTES;

fn peers<R: Runtime>() -> Option<(Client, Client)> {
    let devices = R::enumerate_devices(0);
    for (i, a) in devices.iter().enumerate() {
        for b in &devices[i + 1..] {
            let a = R::client(&R::Device::from_id(*a));
            let b = R::client(&R::Device::from_id(*b));
            if a.peer_access(&b) >= PeerAccess::Memory && b.peer_access(&a) >= PeerAccess::Memory {
                return Some((a, b));
            }
        }
    }
    std::println!("skipping: no two devices bind each other's memory");
    None
}

fn values() -> Vec<u32> {
    (0..BYTES as u32 / 4).collect()
}

fn zeros(client: &Client) -> cubecl_runtime::server::Handle {
    client.create_from_slice(u32::as_bytes(&vec![0u32; BYTES as usize / 4]))
}

pub fn test_peer_load_copies_the_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let peer = remote.create_from_slice(u32::as_bytes(&values()));
    let peer = remote.share_with_peers(&peer).unwrap();
    let out = zeros(&local);
    // Nothing orders the peer's upload before the kernel reading it.
    block_on(remote.sync()).unwrap();

    launch_peer_load(&local, &peer, &out, BYTES);

    let actual = local.read_one(out).unwrap();
    assert_eq!(u32::from_bytes(&actual), values().as_slice());
}

pub fn test_peer_store_copies_into_the_peer_buffer<R: Runtime>() {
    let Some((local, remote)) = peers::<R>() else {
        return;
    };
    let input = local.create_from_slice(u32::as_bytes(&values()));
    let peer = remote.share_with_peers(&zeros(&remote)).unwrap();
    block_on(remote.sync()).unwrap();

    launch_peer_store(&local, &input, &peer, BYTES);
    // Nothing orders the kernel's stores before the owner's read.
    block_on(local.sync()).unwrap();

    let actual = remote.read_one(peer).unwrap();
    assert_eq!(u32::from_bytes(&actual), values().as_slice());
}

pub fn test_a_ring_refuses_one_member_or_a_device_named_twice<R: Runtime>() {
    let Some(&id) = R::enumerate_devices(0).first() else {
        std::println!("skipping: no device");
        return;
    };
    let client = R::client(&R::Device::from_id(id));
    assert!(PeerCopier::hops(core::slice::from_ref(&client)).is_none());
    assert!(PeerCopier::hops(&[client.clone(), client]).is_none());
}

pub fn test_a_ring_sends_each_member_to_the_next_and_the_last_to_the_first<R: Runtime>() {
    let members: Vec<Client> = R::enumerate_devices(0)
        .into_iter()
        .take(3)
        .map(|id| R::client(&R::Device::from_id(id)))
        .collect();
    if members.len() < 2 {
        std::println!("skipping: fewer than two devices");
        return;
    }
    let ids: Vec<_> = members.iter().map(Client::service_id).collect();
    let hops: Vec<_> = PeerCopier::hops(&members)
        .unwrap()
        .map(|(source, destination)| (source.service_id(), destination.service_id()))
        .collect();
    let next = (0..ids.len()).map(|index| (ids[index], ids[(index + 1) % ids.len()]));
    assert_eq!(hops, next.collect::<Vec<_>>());
}

#[macro_export]
macro_rules! testgen_link {
    () => {
        mod link {
            use super::*;

            #[$crate::tests::test_log::test]
            fn peer_load_copies_the_peer_buffer() {
                cubecl_std::tests::link::test_peer_load_copies_the_peer_buffer::<TestRuntime>();
            }

            #[$crate::tests::test_log::test]
            fn peer_store_copies_into_the_peer_buffer() {
                cubecl_std::tests::link::test_peer_store_copies_into_the_peer_buffer::<TestRuntime>(
                );
            }

            #[$crate::tests::test_log::test]
            fn a_ring_refuses_one_member_or_a_device_named_twice() {
                cubecl_std::tests::link::test_a_ring_refuses_one_member_or_a_device_named_twice::<
                    TestRuntime,
                >();
            }

            #[$crate::tests::test_log::test]
            fn a_ring_sends_each_member_to_the_next_and_the_last_to_the_first() {
                cubecl_std::tests::link::test_a_ring_sends_each_member_to_the_next_and_the_last_to_the_first::<TestRuntime>();
            }
        }
    };
}

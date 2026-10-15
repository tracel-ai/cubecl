use crate::prelude::*;
use alloc::{vec, vec::Vec};
use core::time::Duration;
use cubecl_common::device::{Device, DeviceId};
use cubecl_runtime::{
    client::{Client, ClientGroup, ClientGroupError, MemberBuffers},
    runtime::Runtime,
    server::{Handle, ReduceOperation},
};
use std::sync::{
    Mutex, MutexGuard, PoisonError,
    mpsc::{self, RecvTimeoutError},
};

/// Tests in one process share each device's communicators, which pair collectives in the order
/// each device queues them, so the tests queue theirs one at a time.
static COLLECTIVES: Mutex<()> = Mutex::new(());

/// Devices whose communicators were queued in different orders hang rather than fail.
const TIME_LIMIT: Duration = Duration::from_secs(60);

const F32: cubecl_ir::ElemType = cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32);

/// Elements of each member's contribution, and of each chunk.
const LEN: usize = 6;

/// No collective here writes it, since no contribution is negative.
const UNWRITTEN: f32 = -1.0;

pub fn test_a_group_all_reduce_sums_and_averages<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(all_reduce::<R>);
}

pub fn test_all_gather_concatenates_in_member_order<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(all_gather::<R>);
}

pub fn test_reduce_scatter_gives_each_member_its_chunk<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(reduce_scatter::<R>);
}

pub fn test_broadcast_in_place_copies_the_root_to_every_member<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(broadcast::<R>);
}

pub fn test_all_to_all_exchanges_chunks_by_rank<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(all_to_all::<R>);
}

pub fn test_a_group_without_the_first_device_gathers<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(later_devices::<R>);
}

pub fn test_a_transfer_moves_bytes_of_any_type<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(transfer_of_any_type::<R>);
}

pub fn test_a_collective_refuses_a_repeated_device<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(repeated_device::<R>);
}

pub fn test_a_group_refuses_mismatched_buffers_before_any_member_queues<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(mismatched_buffers::<R>);
}

pub fn test_a_broadcast_refuses_a_source_on_no_member<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(source_on_no_member::<R>);
}

/// Holds every device's collectives for the caller until the guard drops.
pub fn queue_alone() -> MutexGuard<'static, ()> {
    COLLECTIVES.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `test`, aborting the process if it hangs.
pub fn finish_within_limit(test: fn()) {
    let (finished, done) = mpsc::channel();
    let test = std::thread::spawn(move || {
        test();
        let _ = finished.send(());
    });
    if let Err(RecvTimeoutError::Timeout) = done.recv_timeout(TIME_LIMIT) {
        // A panic would leave the devices waiting on each other for every later test.
        std::eprintln!(
            "still running after {TIME_LIMIT:?}, most likely on devices waiting for each other"
        );
        std::process::abort();
    }
    if let Err(panic) = test.join() {
        std::panic::resume_unwind(panic);
    }
}

fn all_reduce<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let contributions = || -> Vec<MemberBuffers> {
        group
            .clients()
            .iter()
            .enumerate()
            .map(|(member, client)| MemberBuffers {
                src: from_values(client, &values(member, LEN)),
                dst: unwritten(client, LEN),
            })
            .collect()
    };
    let summed: Vec<f32> = (0..LEN)
        .map(|i| (0..members).map(|member| values(member, LEN)[i]).sum())
        .collect();

    let buffers = contributions();
    let sums = destinations(&buffers);
    group.all_reduce(buffers, F32, ReduceOperation::Sum);
    assert_every_member_holds(&group, sums, |_| summed.clone());

    let buffers = contributions();
    let means = destinations(&buffers);
    group.all_reduce(buffers, F32, ReduceOperation::Mean);
    for (member, (client, destination)) in group.clients().iter().zip(means).enumerate() {
        let actual = client.read_one(destination).unwrap();
        for (actual, sum) in f32::from_bytes(&actual).iter().zip(&summed) {
            let mean = sum / members as f32;
            assert!((actual - mean).abs() <= mean * 1e-6, "member {member}");
        }
    }
}

fn all_gather<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    gather_every_member(&group);
}

fn reduce_scatter<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let buffers: Vec<_> = group
        .clients()
        .iter()
        .enumerate()
        .map(|(member, client)| MemberBuffers {
            src: from_values(client, &values(member, members * LEN)),
            dst: unwritten(client, LEN),
        })
        .collect();
    let destinations = destinations(&buffers);

    group.reduce_scatter(buffers, F32, ReduceOperation::Sum);

    let reduced: Vec<f32> = (0..members * LEN)
        .map(|i| {
            (0..members)
                .map(|member| values(member, members * LEN)[i])
                .sum()
        })
        .collect();
    assert_every_member_holds(&group, destinations, |member| {
        reduced[member * LEN..(member + 1) * LEN].to_vec()
    });
}

fn broadcast<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    // The last member, so a root taken for rank 0 shows.
    let root = group.clients().len() - 1;
    let destinations: Vec<_> = group
        .clients()
        .iter()
        .enumerate()
        .map(|(member, client)| match member == root {
            true => from_values(client, &values(root, LEN)),
            false => unwritten(client, LEN),
        })
        .collect();

    group.broadcast(destinations[root].clone(), destinations.clone());

    assert_every_member_holds(&group, destinations, |_| values(root, LEN));
}

fn all_to_all<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let chunk = |from: usize, to: usize| -> Vec<f32> {
        (0..LEN)
            .map(|i| (100 * from + 10 * to + i) as f32)
            .collect()
    };
    let buffers: Vec<_> = group
        .clients()
        .iter()
        .enumerate()
        .map(|(member, client)| {
            let src: Vec<f32> = (0..members).flat_map(|to| chunk(member, to)).collect();
            MemberBuffers {
                src: from_values(client, &src),
                dst: unwritten(client, members * LEN),
            }
        })
        .collect();
    let destinations = destinations(&buffers);

    group.all_to_all(buffers);

    assert_every_member_holds(&group, destinations, |member| {
        (0..members).flat_map(|from| chunk(from, member)).collect()
    });
}

fn later_devices<R: Runtime>() {
    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return;
    }
    let Some(group) = group_of::<R>(&device_ids[1..]) else {
        return;
    };
    gather_every_member(&group);
}

fn transfer_of_any_type<R: Runtime>() {
    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return;
    }
    let mut source = R::client(&R::Device::from_id(device_ids[0]));
    let destination = R::client(&R::Device::from_id(device_ids[1]));
    if !source.has_device_transport() {
        log::warn!("skipping transfers: {} has no transport", source.name());
        return;
    }
    // Three `i16`s, a type NCCL has no name for.
    let bytes = [1u8, 2, 3, 4, 5, 6];
    let src = source.create_from_slice(&bytes);

    let moved = source.to_client(src, &destination);

    assert_eq!(destination.read_one(moved).unwrap().to_vec(), bytes);
}

fn repeated_device<R: Runtime>() {
    let Some(device) = R::enumerate_devices(0).first().copied() else {
        return;
    };
    let client = R::client(&R::Device::from_id(device));
    if !client.has_device_transport() {
        return;
    }

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.ensure_init_collective(vec![device, device]);
    }));

    assert_panicked_with(refused, "appears twice");
}

fn mismatched_buffers<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let gather = |last_dst: usize| -> Vec<MemberBuffers> {
        group
            .clients()
            .iter()
            .enumerate()
            .map(|(member, client)| MemberBuffers {
                src: from_values(client, &values(member, LEN)),
                dst: unwritten(
                    client,
                    match member == members - 1 {
                        true => last_dst,
                        false => members * LEN,
                    },
                ),
            })
            .collect()
    };
    let refused = |reason: &str, collective: &dyn Fn()| {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(collective));
        assert_panicked_with(outcome, reason);
    };

    refused("holds", &|| group.all_gather(gather(LEN)));
    assert!(group.reduces(F32));
    assert!(!group.reduces(cubecl_ir::ElemType::Bool));
    refused("cannot reduce", &|| {
        group.all_reduce(gather(LEN), cubecl_ir::ElemType::Bool, ReduceOperation::Sum)
    });
    refused("two streams", &|| {
        let mut two_streams = gather(members * LEN);
        let client = group.clients()[0].clone();
        two_streams[0].src = std::thread::spawn(move || from_values(&client, &values(0, LEN)))
            .join()
            .unwrap();
        group.all_gather(two_streams)
    });
    refused("overlap", &|| {
        let overlapping = group
            .clients()
            .iter()
            .map(|client| {
                let dst = unwritten(client, members * LEN);
                let first_chunk = (members - 1) * LEN * size_of::<f32>();
                MemberBuffers {
                    src: dst.clone().offset_end(first_chunk as u64),
                    dst,
                }
            })
            .collect();
        group.all_gather(overlapping)
    });
    refused("whole", &|| {
        let partial = group
            .clients()
            .iter()
            .map(|client| MemberBuffers {
                src: client.create_from_slice(&[0; 2 + LEN * size_of::<f32>()]),
                dst: client.create_from_slice(&[0; 2 + LEN * size_of::<f32>()]),
            })
            .collect();
        group.all_reduce(partial, F32, ReduceOperation::Sum)
    });
    refused("evenly", &|| {
        let uneven = group
            .clients()
            .iter()
            .map(|client| MemberBuffers {
                src: client.create_from_slice(&vec![0; 1 + members * LEN * size_of::<f32>()]),
                dst: unwritten(client, LEN),
            })
            .collect();
        group.reduce_scatter(uneven, F32, ReduceOperation::Sum)
    });

    // No member queued a part, so the devices still run the next collective.
    gather_every_member(&group);
}

fn source_on_no_member<R: Runtime>() {
    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return;
    }
    let Some(group) = group_of::<R>(&device_ids[1..]) else {
        return;
    };
    let outsider = R::client(&R::Device::from_id(device_ids[0]));
    let src = from_values(&outsider, &values(0, LEN));
    let destinations = group
        .clients()
        .iter()
        .map(|client| unwritten(client, LEN))
        .collect();

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        group.broadcast(src, destinations)
    }));

    assert_panicked_with(refused, "no member's device");
}

/// Gathers `values(member, LEN)` from every member, checking each holds them all in order.
fn gather_every_member(group: &ClientGroup) {
    let members = group.clients().len();
    let buffers: Vec<_> = group
        .clients()
        .iter()
        .enumerate()
        .map(|(member, client)| MemberBuffers {
            src: from_values(client, &values(member, LEN)),
            dst: unwritten(client, members * LEN),
        })
        .collect();
    let destinations = destinations(&buffers);

    group.all_gather(buffers);

    let expected: Vec<f32> = (0..members)
        .flat_map(|member| values(member, LEN))
        .collect();
    assert_every_member_holds(group, destinations, |_| expected.clone());
}

fn assert_panicked_with(outcome: std::thread::Result<()>, reason: &str) {
    let panic = outcome.expect_err("the collective was accepted");
    let message = panic
        .downcast_ref::<std::string::String>()
        .map(std::string::String::as_str)
        .unwrap_or_default();
    assert!(message.contains(reason), "{message}");
}

/// A group of every device, or `None` when there are fewer than two or no transport.
fn group<R: Runtime>() -> Option<ClientGroup> {
    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return None;
    }
    group_of::<R>(&device_ids)
}

/// A group of `device_ids`, or `None` when their runtime has no transport.
fn group_of<R: Runtime>(device_ids: &[DeviceId]) -> Option<ClientGroup> {
    let clients = device_ids
        .iter()
        .map(|id| R::client(&R::Device::from_id(*id)))
        .collect();
    match ClientGroup::new(clients) {
        Ok(group) => Some(group),
        Err(ClientGroupError::NoDeviceTransport { runtime }) => {
            log::warn!("skipping collectives: {runtime} has no transport between its devices");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

/// `len` values that tell `member`'s contribution apart from every other.
fn values(member: usize, len: usize) -> Vec<f32> {
    (0..len).map(|i| (100 * member + i) as f32).collect()
}

fn from_values(client: &Client, values: &[f32]) -> Handle {
    client.create_from_slice(f32::as_bytes(values))
}

/// A destination of `len` values, so one read before its result lands shows.
fn unwritten(client: &Client, len: usize) -> Handle {
    from_values(client, &vec![UNWRITTEN; len])
}

fn destinations(buffers: &[MemberBuffers]) -> Vec<Handle> {
    buffers.iter().map(|buffers| buffers.dst.clone()).collect()
}

fn assert_every_member_holds(
    group: &ClientGroup,
    destinations: Vec<Handle>,
    expected: impl Fn(usize) -> Vec<f32>,
) {
    for (member, (client, destination)) in group.clients().iter().zip(destinations).enumerate() {
        let actual = client.read_one(destination).unwrap();
        assert_eq!(
            f32::from_bytes(&actual),
            expected(member),
            "member {member}"
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_collectives {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_a_group_all_reduce_sums_and_averages() {
            cubecl_core::runtime_tests::collectives::test_a_group_all_reduce_sums_and_averages::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_gather_concatenates_in_member_order() {
            cubecl_core::runtime_tests::collectives::test_all_gather_concatenates_in_member_order::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_reduce_scatter_gives_each_member_its_chunk() {
            cubecl_core::runtime_tests::collectives::test_reduce_scatter_gives_each_member_its_chunk::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_broadcast_in_place_copies_the_root_to_every_member() {
            cubecl_core::runtime_tests::collectives::test_broadcast_in_place_copies_the_root_to_every_member::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_to_all_exchanges_chunks_by_rank() {
            cubecl_core::runtime_tests::collectives::test_all_to_all_exchanges_chunks_by_rank::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_group_without_the_first_device_gathers() {
            cubecl_core::runtime_tests::collectives::test_a_group_without_the_first_device_gathers::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_transfer_moves_bytes_of_any_type() {
            cubecl_core::runtime_tests::collectives::test_a_transfer_moves_bytes_of_any_type::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_group_refuses_mismatched_buffers_before_any_member_queues() {
            cubecl_core::runtime_tests::collectives::test_a_group_refuses_mismatched_buffers_before_any_member_queues::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_broadcast_refuses_a_source_on_no_member() {
            cubecl_core::runtime_tests::collectives::test_a_broadcast_refuses_a_source_on_no_member::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_collective_refuses_a_repeated_device() {
            cubecl_core::runtime_tests::collectives::test_a_collective_refuses_a_repeated_device::<
                TestRuntime,
            >();
        }
    };
}

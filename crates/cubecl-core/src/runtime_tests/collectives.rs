use crate::prelude::*;
use alloc::{vec, vec::Vec};
use core::time::Duration;
use cubecl_common::device::Device;
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

pub fn test_a_group_all_reduce_sums_every_member<R: Runtime>() {
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

pub fn test_broadcast_copies_the_root_to_every_member<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(broadcast::<R>);
}

pub fn test_all_to_all_exchanges_chunks_by_rank<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(all_to_all::<R>);
}

pub fn test_a_collective_refuses_a_repeated_device<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(repeated_device::<R>);
}

pub fn test_a_group_refuses_mismatched_buffers_before_any_member_queues<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(mismatched_buffers::<R>);
}

pub fn test_a_broadcast_takes_a_source_on_the_root_only<R: Runtime>() {
    let _collectives = queue_alone();
    finish_within_limit(source_off_the_root::<R>);
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
    let Some(mut group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let buffers: Vec<_> = group
        .clients()
        .iter()
        .enumerate()
        .map(|(member, client)| MemberBuffers {
            src: from_values(client, &values(member, LEN)),
            dst: unwritten(client, LEN),
        })
        .collect();
    let destinations = destinations(&buffers);

    group.all_reduce(buffers, F32, ReduceOperation::Sum);

    let summed: Vec<f32> = (0..LEN)
        .map(|i| (0..members).map(|member| values(member, LEN)[i]).sum())
        .collect();
    assert_every_member_holds(&group, destinations, |_| summed.clone());
}

fn all_gather<R: Runtime>() {
    let Some(mut group) = group::<R>() else {
        return;
    };
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

    group.all_gather(buffers, F32);

    let expected: Vec<f32> = (0..members)
        .flat_map(|member| values(member, LEN))
        .collect();
    assert_every_member_holds(&group, destinations, |_| expected.clone());
}

fn reduce_scatter<R: Runtime>() {
    let Some(mut group) = group::<R>() else {
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
    let Some(mut group) = group::<R>() else {
        return;
    };
    // The last member, so a root taken for rank 0 shows.
    let root = group.clients().len() - 1;
    let src = from_values(&group.clients()[root], &values(root, LEN));
    let destinations: Vec<_> = group
        .clients()
        .iter()
        .map(|client| unwritten(client, LEN))
        .collect();

    group.broadcast(root, src, destinations.clone(), F32);

    assert_every_member_holds(&group, destinations, |_| values(root, LEN));
}

fn all_to_all<R: Runtime>() {
    let Some(mut group) = group::<R>() else {
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

    group.all_to_all(buffers, F32);

    assert_every_member_holds(&group, destinations, |member| {
        (0..members).flat_map(|from| chunk(from, member)).collect()
    });
}

fn repeated_device<R: Runtime>() {
    let Some(group) = group::<R>() else {
        return;
    };
    let mut client = group.clients()[0].clone();
    let device = R::enumerate_devices(0)[0];

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.ensure_init_collective(vec![device, device]);
    }));

    assert_panicked_with(refused, "appears twice");
}

fn mismatched_buffers<R: Runtime>() {
    let Some(mut group) = group::<R>() else {
        return;
    };
    let members = group.clients().len();
    let gather = |group: &ClientGroup, last_dst: usize| -> Vec<MemberBuffers> {
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

    let wrong_size = gather(&group, LEN);
    assert_refused(&mut group, "holds", |group| {
        group.all_gather(wrong_size, F32)
    });
    let features = &group.clients()[0].properties().features;
    assert!(features.device_transport_moves(F32));
    let untaken = gather(&group, members * LEN);
    assert_refused(&mut group, "cannot move", |group| {
        group.all_gather(untaken, cubecl_ir::ElemType::Bool)
    });
    let mut two_streams = gather(&group, members * LEN);
    let client = group.clients()[0].clone();
    two_streams[0].src = std::thread::spawn(move || from_values(&client, &values(0, LEN)))
        .join()
        .unwrap();
    assert_refused(&mut group, "two streams", |group| {
        group.all_gather(two_streams, F32)
    });
    let partial = group
        .clients()
        .iter()
        .map(|client| MemberBuffers {
            src: client.create_from_slice(&[0; 2 + LEN * size_of::<f32>()]),
            dst: unwritten(client, members * LEN),
        })
        .collect();
    assert_refused(&mut group, "whole", |group| group.all_gather(partial, F32));

    // No member queued a part, so the devices still run the next collective.
    let buffers = gather(&group, members * LEN);
    let destinations = destinations(&buffers);
    group.all_gather(buffers, F32);
    let expected: Vec<f32> = (0..members)
        .flat_map(|member| values(member, LEN))
        .collect();
    assert_every_member_holds(&group, destinations, |_| expected.clone());
}

fn source_off_the_root<R: Runtime>() {
    let Some(mut group) = group::<R>() else {
        return;
    };
    let src = from_values(&group.clients()[0], &values(0, LEN));
    let destinations = group
        .clients()
        .iter()
        .map(|client| unwritten(client, LEN))
        .collect();

    assert_refused(&mut group, "not on its device", |group| {
        group.broadcast(1, src, destinations, F32)
    });
}

fn assert_refused(
    group: &mut ClientGroup,
    reason: &str,
    collective: impl FnOnce(&mut ClientGroup),
) {
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| collective(group)));
    assert_panicked_with(refused, reason);
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
        fn test_a_group_all_reduce_sums_every_member() {
            cubecl_core::runtime_tests::collectives::test_a_group_all_reduce_sums_every_member::<
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
        fn test_broadcast_copies_the_root_to_every_member() {
            cubecl_core::runtime_tests::collectives::test_broadcast_copies_the_root_to_every_member::<
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
        fn test_a_group_refuses_mismatched_buffers_before_any_member_queues() {
            cubecl_core::runtime_tests::collectives::test_a_group_refuses_mismatched_buffers_before_any_member_queues::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_a_broadcast_takes_a_source_on_the_root_only() {
            cubecl_core::runtime_tests::collectives::test_a_broadcast_takes_a_source_on_the_root_only::<
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

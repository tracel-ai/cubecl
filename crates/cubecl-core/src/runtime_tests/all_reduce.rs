use crate::prelude::*;
use alloc::vec::Vec;
use core::time::Duration;
use cubecl_common::device::{Device, DeviceId};
use cubecl_runtime::{client::Client, runtime::Runtime};
use std::sync::{
    Mutex, PoisonError,
    mpsc::{self, RecvTimeoutError},
};

/// Tests in one process share each device's communicators, which pair `all_reduce` calls in the
/// order each device queues them, so the tests queue theirs one at a time.
static COLLECTIVES: Mutex<()> = Mutex::new(());

/// Devices whose communicators were queued in different orders hang rather than fail.
const TIME_LIMIT: Duration = Duration::from_secs(60);

const F32: cubecl_ir::ElemType = cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32);

pub fn test_all_reduce_sync_collective<R: Runtime>() {
    let _collectives = COLLECTIVES.lock().unwrap_or_else(PoisonError::into_inner);
    finish_within_limit(sync_collective::<R>);
}

/// Transfers between two devices, beside `all_reduce` calls queued over them device by device as
/// a gradient sync does, keep one order on both devices, so neither hangs.
pub fn test_all_reduce_beside_transfers<R: Runtime>() {
    let _collectives = COLLECTIVES.lock().unwrap_or_else(PoisonError::into_inner);
    finish_within_limit(beside_transfers::<R>);
}

/// A group's first `all_reduce` calls finish when one thread queues more of them on each device
/// than its task queue holds before moving to the next device. It is a first use only when no
/// test before it in the process set the group up.
pub fn test_all_reduce_first_use_many_parts<R: Runtime>() {
    let _collectives = COLLECTIVES.lock().unwrap_or_else(PoisonError::into_inner);
    finish_within_limit(first_use_many_parts::<R>);
}

/// Tensor parallel groups over devices 0 and 1 and over 2 and 3, a thread per device, keep their
/// `all_reduce` calls and the pipeline transfers between the groups in one order, whichever comes
/// first on a rank, and so does a data parallel `all_reduce` over all four.
pub fn test_all_reduce_tensor_and_pipeline<R: Runtime>() {
    let _collectives = COLLECTIVES.lock().unwrap_or_else(PoisonError::into_inner);
    finish_within_limit(tensor_and_pipeline::<R>);
}

fn finish_within_limit(test: fn()) {
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

fn sync_collective<R: Runtime>() {
    let type_id = 0;
    let device_ids = R::enumerate_devices(type_id);
    let device_count = device_ids.len();

    if device_count < 2 {
        return;
    }
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id))
        .collect();
    if !R::client(&devices[0]).has_device_transport() {
        log::warn!(
            "skipping all_reduce: {} devices but no transport between them",
            device_count
        );
        return;
    }

    const SIZE: usize = 100;
    const NUM_HANDLES: usize = 8;

    let mut jobs = devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let client = R::client(device);
            let handles = (0..NUM_HANDLES)
                .map(|j| {
                    let src = [i as f32 + j as f32; SIZE];
                    client.create_from_slice(f32::as_bytes(&src))
                })
                .collect::<Vec<_>>();
            (client, handles)
        })
        .collect::<Vec<_>>();

    for (client, handles) in jobs.iter_mut() {
        for handle in handles.iter() {
            client.all_reduce(
                handle.clone(),
                handle.clone(),
                F32,
                device_ids.clone(),
                cubecl_runtime::server::ReduceOperation::Sum,
            );
        }

        // We perform the collective sync AFTER all all_reduce calls.
        client.sync_collective();
    }

    let value_base: f32 = device_ids.iter().map(|id| id.index_id as f32).sum();

    for (client, handles) in jobs.into_iter() {
        for (j, handle) in handles.into_iter().enumerate() {
            let actual = client.read_one(handle).unwrap();
            let actual = f32::from_bytes(&actual);
            let expected = [value_base + j as f32 * device_count as f32; SIZE];
            assert_eq!(actual, expected);
        }
    }
}

fn first_use_many_parts<R: Runtime>() {
    const PARTS: usize = 100;
    const SIZE: usize = 16;
    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return;
    }
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id))
        .collect();
    if !R::client(&devices[0]).has_device_transport() {
        return;
    }

    let mut clients: Vec<_> = devices.iter().map(R::client).collect();
    let handles: Vec<Vec<_>> = clients
        .iter()
        .enumerate()
        .map(|(rank, client)| {
            (0..PARTS)
                .map(|_| client.create_from_slice(f32::as_bytes(&[rank as f32; SIZE])))
                .collect()
        })
        .collect();
    for (client, handles) in clients.iter_mut().zip(&handles) {
        for handle in handles {
            client.all_reduce(
                handle.clone(),
                handle.clone(),
                F32,
                device_ids.clone(),
                cubecl_runtime::server::ReduceOperation::Sum,
            );
        }
    }
    for client in &clients {
        client.sync_collective();
    }

    let ranks = clients.len();
    let expected = [(ranks * (ranks - 1) / 2) as f32; SIZE];
    for (client, handles) in clients.iter().zip(handles) {
        for handle in handles {
            let actual = client.read_one(handle).unwrap();
            assert_eq!(f32::from_bytes(&actual), expected);
        }
    }
}

fn tensor_and_pipeline<R: Runtime>() {
    const ROUNDS: usize = 20;

    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 4 {
        return;
    }
    let device_ids = device_ids[..4].to_vec();
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id))
        .collect();
    if !R::client(&devices[0]).has_device_transport() {
        return;
    }

    std::thread::scope(|scope| {
        for rank in 0..4 {
            let (devices, device_ids) = (&devices, &device_ids);
            scope.spawn(move || {
                let mut client = R::client(&devices[rank]);
                let next_stage = R::client(&devices[(rank + 2) % 4]);
                let stage = rank / 2 * 2..rank / 2 * 2 + 2;

                for round in 0..ROUNDS {
                    let value = |rank: usize| (rank + round) as f32;
                    let transfer = |client: &mut Client| {
                        let sent = [(100 * rank + round) as f32; 64];
                        let input = client.create_from_slice(f32::as_bytes(&sent));
                        let output = client.to_client(input, &next_stage, F32);
                        let received = next_stage.read_one_unchecked(output);
                        assert_eq!(
                            f32::from_bytes(&received),
                            sent,
                            "rank {rank} round {round}"
                        );
                    };
                    // The two ranks of a stage take these in opposite orders, so a rank often
                    // transfers while its partner's part is already queued.
                    if (rank + round) % 2 == 0 {
                        transfer(&mut client);
                    }
                    let tensor_parallel: f32 = stage.clone().map(value).sum();
                    assert_eq!(
                        reduced(&mut client, &device_ids[stage.clone()], value(rank)),
                        [tensor_parallel; 64]
                    );
                    if (rank + round) % 2 == 1 {
                        transfer(&mut client);
                    }
                    let data_parallel: f32 = (0..4).map(value).sum();
                    assert_eq!(
                        reduced(&mut client, device_ids, value(rank)),
                        [data_parallel; 64]
                    );
                }
            });
        }
    });
}

/// `value` from `client`'s device, summed over `group`.
fn reduced(client: &mut Client, group: &[DeviceId], value: f32) -> Vec<f32> {
    let handle = client.create_from_slice(f32::as_bytes(&[value; 64]));
    client.all_reduce(
        handle.clone(),
        handle.clone(),
        F32,
        group.to_vec(),
        cubecl_runtime::server::ReduceOperation::Sum,
    );
    client.sync_collective();
    f32::from_bytes(&client.read_one(handle).unwrap()).to_vec()
}

fn beside_transfers<R: Runtime>() {
    const ROUNDS: usize = 20;
    const HANDLES: usize = 8;
    const SIZE: usize = 64;

    let device_ids = R::enumerate_devices(0);
    if device_ids.len() < 2 {
        return;
    }
    let device_ids = device_ids[..2].to_vec();
    let devices: Vec<R::Device> = device_ids
        .iter()
        .map(|id| R::Device::from_id(*id))
        .collect();
    if !R::client(&devices[0]).has_device_transport() {
        return;
    }

    std::thread::scope(|scope| {
        for thread in 0..2 {
            let (source, destination) = match thread {
                0 => (&devices[0], &devices[1]),
                _ => (&devices[1], &devices[0]),
            };
            scope.spawn(move || {
                let mut source = R::client(source);
                let destination = R::client(destination);

                for round in 0..ROUNDS {
                    let expected = [(thread * ROUNDS + round) as f32; SIZE];
                    let input = source.create_from_slice(f32::as_bytes(&expected));
                    let output = source.to_client(input, &destination, F32);
                    let actual = destination.read_one_unchecked(output);
                    assert_eq!(f32::from_bytes(&actual), expected);
                }
            });
        }

        let (devices, device_ids) = (&devices, &device_ids);
        scope.spawn(move || {
            let mut clients: Vec<_> = devices.iter().map(R::client).collect();

            for round in 0..ROUNDS {
                let handles: Vec<Vec<_>> = clients
                    .iter()
                    .enumerate()
                    .map(|(rank, client)| {
                        (0..HANDLES)
                            .map(|_| {
                                let values = [(rank + round) as f32; SIZE];
                                client.create_from_slice(f32::as_bytes(&values))
                            })
                            .collect()
                    })
                    .collect();

                for (client, handles) in clients.iter_mut().zip(&handles) {
                    for handle in handles {
                        client.all_reduce(
                            handle.clone(),
                            handle.clone(),
                            F32,
                            device_ids.clone(),
                            cubecl_runtime::server::ReduceOperation::Sum,
                        );
                    }
                }
                for client in &clients {
                    client.sync_collective();
                }

                let expected = [(2 * round + 1) as f32; SIZE];
                for (client, handles) in clients.iter().zip(handles) {
                    for handle in handles {
                        let actual = client.read_one(handle).unwrap();
                        assert_eq!(f32::from_bytes(&actual), expected);
                    }
                }
            }
        });
    });
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all_reduce {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_sync_collective() {
            cubecl_core::runtime_tests::all_reduce::test_all_reduce_sync_collective::<TestRuntime>(
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_first_use_many_parts() {
            cubecl_core::runtime_tests::all_reduce::test_all_reduce_first_use_many_parts::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_tensor_and_pipeline() {
            cubecl_core::runtime_tests::all_reduce::test_all_reduce_tensor_and_pipeline::<
                TestRuntime,
            >();
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_all_reduce_beside_transfers() {
            cubecl_core::runtime_tests::all_reduce::test_all_reduce_beside_transfers::<TestRuntime>(
            );
        }
    };
}

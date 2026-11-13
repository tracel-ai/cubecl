use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cubecl::{
    client::{ClientGroup, MemberBuffers},
    future::block_on,
    ir::{ElemType, FloatKind},
    logging::{
        Duration, Execution, ExecutionId, ExecutionKind, ExecutionObservation, ExecutionObserver,
        Measurement, ProfileDuration, Timed, TimingRequest,
    },
    prelude::*,
    server::ReduceOperation,
};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A decode step's hidden state at batch one.
const HIDDEN: usize = 2048;

/// Elements the spinning kernel holds back the late member with, one per unit.
const SPUN: usize = 1 << 20;
const SPINS: u32 = 4096;

const ROUNDS: usize = 3;

#[cube(launch_unchecked)]
fn spin(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let mut x = output[ABSOLUTE_POS];
        for _ in 0..SPINS {
            x = x * 1.0001 + 1.0;
        }
        output[ABSOLUTE_POS] = x;
    }
}

/// Every execution of one round, and its measurement once read back.
#[derive(Default)]
struct Round {
    issued: Mutex<Vec<Execution>>,
    unread: Mutex<Vec<(ExecutionId, ProfileDuration)>>,
}

impl ExecutionObserver for Round {
    fn issued(&self, execution: &Execution) {
        self.issued.lock().unwrap().push(execution.clone());
    }

    fn timing(&self) -> TimingRequest {
        TimingRequest::Deferred(Timed::KernelsAndCommunications)
    }

    fn measured(&self, execution: &Execution, measurement: Measurement) {
        if let Measurement::Unread(measurement) = measurement {
            self.unread
                .lock()
                .unwrap()
                .push((execution.id, measurement));
        }
    }
}

impl Round {
    /// The time the late member was held back, then each member's part and wait.
    fn report(&self, group: &ClientGroup) -> String {
        let read: HashMap<ExecutionId, Duration> =
            std::mem::take(&mut *self.unread.lock().unwrap())
                .into_iter()
                .filter_map(|(id, measurement)| {
                    Some((id, block_on(measurement.resolve())?.duration()))
                })
                .collect();
        let issued = self.issued.lock().unwrap();
        let time = |device, wanted: fn(&ExecutionKind) -> bool| -> Duration {
            issued
                .iter()
                .filter(|execution| {
                    wanted(&execution.kind) && placed_on(&execution.kind) == Some(device)
                })
                .filter_map(|execution| read.get(&execution.id))
                .sum()
        };
        let late = group.clients().last().unwrap().service_id().device;
        let mut line = format!(
            "held back {:>10.1?}",
            time(late, |kind| matches!(kind, ExecutionKind::Kernel { .. }))
        );
        for (member, client) in group.clients().iter().enumerate() {
            let device = client.service_id().device;
            line += &format!(
                " | member {member}: part {:>10.1?} wait {:>10.1?}",
                time(device, |kind| matches!(kind, ExecutionKind::Part { .. })),
                time(device, |kind| matches!(kind, ExecutionKind::Wait { .. })),
            );
        }
        line
    }
}

fn placed_on(kind: &ExecutionKind) -> Option<cubecl::device::DeviceId> {
    match kind {
        ExecutionKind::Kernel { place, .. }
        | ExecutionKind::Part { place }
        | ExecutionKind::Wait { place, .. } => Some(place.service.device),
        ExecutionKind::Communication(_) => None,
    }
}

fn buffers(group: &ClientGroup) -> Vec<MemberBuffers> {
    group
        .clients()
        .iter()
        .map(|client| {
            let buffer = client.empty(HIDDEN * F32.size());
            MemberBuffers {
                src: buffer.clone(),
                dst: buffer,
            }
        })
        .collect()
}

fn sync(group: &ClientGroup) {
    for client in group.clients() {
        block_on(client.sync()).expect("the member syncs");
    }
}

fn main() {
    let group = ClientGroup::new(multi_device::clients()).expect("the devices form a group");
    let late = group.clients().last().unwrap();
    let spun = late.empty(SPUN * F32.size());

    // Connects the group and compiles the kernel, so no round measures either.
    group.all_reduce(buffers(&group), F32, ReduceOperation::Sum);
    hold_back(late, &spun, 1);
    sync(&group);

    println!("all-reduce of [1, {HIDDEN}] f32, the last member held back by spinning kernels");
    for launches in [0, 4, 16] {
        for _ in 0..ROUNDS {
            let buffers = buffers(&group);
            let round = Arc::new(Round::default());
            let watching = ExecutionObservation::new(round.clone());
            hold_back(late, &spun, launches);
            group.all_reduce(buffers, F32, ReduceOperation::Sum);
            drop(watching);
            sync(&group);
            println!("{launches:>3} launches: {}", round.report(&group));
        }
    }

    println!("the same all-reduce waited for at a sync every member reaches idle, after it ended");
    let spun: Vec<_> = group
        .clients()
        .iter()
        .map(|client| client.empty(SPUN * F32.size()))
        .collect();
    let devices: Vec<_> = group
        .clients()
        .iter()
        .map(|client| client.service_id().device)
        .collect();
    for _ in 0..ROUNDS {
        let buffers = buffers(&group);
        let round = Arc::new(Round::default());
        let watching = ExecutionObservation::new(round.clone());
        for (client, buffers) in group.clients().iter().zip(&buffers) {
            client.clone().all_reduce(
                buffers.src.clone(),
                buffers.dst.clone(),
                F32,
                devices.clone(),
                ReduceOperation::Sum,
            );
        }
        // Each compute stream runs past the collective, which waits for none of them.
        for (client, spun) in group.clients().iter().zip(&spun) {
            hold_back(client, spun, 16);
        }
        sync(&group);
        for client in group.clients() {
            client.sync_collective();
        }
        drop(watching);
        sync(&group);
        println!("    at sync: {}", round.report(&group));
    }
}

fn hold_back(client: &Client, spun: &cubecl::server::Handle, launches: usize) {
    for _ in 0..launches {
        unsafe {
            spin::launch_unchecked(
                client,
                CubeCount::Static((SPUN as u32).div_ceil(256), 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(spun.clone(), SPUN),
            );
        }
    }
}

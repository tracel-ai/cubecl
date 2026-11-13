use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cubecl::{
    client::{ClientGroup, MemberBuffers},
    future::block_on,
    ir::{ElemType, FloatKind},
    logging::{
        Execution, ExecutionObservation, ExecutionObserver, Measurement, ProfileDuration, Timed,
        TimingRequest,
    },
    prelude::*,
    server::{Handle, ReduceOperation},
};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A decode step's hidden state at batch one.
const HIDDEN: usize = 2048;

/// Collectives in one pass, each after a kernel on every member, as a decode step's layers.
const STEPS: usize = 256;

const REPS: usize = 5;

#[cube(launch_unchecked)]
fn layer(state: &mut [f32]) {
    if ABSOLUTE_POS < state.len() {
        let mut x = state[ABSOLUTE_POS];
        for _ in 0..64u32 {
            x = x * 0.999 + 0.001;
        }
        state[ABSOLUTE_POS] = x;
    }
}

/// Keeps every measurement unread until the pass is over, as a profiler would.
struct Keeps {
    timing: TimingRequest,
    unread: Mutex<Vec<ProfileDuration>>,
}

impl ExecutionObserver for Keeps {
    fn issued(&self, _execution: &Execution) {}

    fn timing(&self) -> TimingRequest {
        self.timing
    }

    fn measured(&self, _execution: &Execution, measurement: Measurement) {
        if let Measurement::Unread(measurement) = measurement {
            self.unread.lock().unwrap().push(measurement);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Observed {
    Not,
    Kernels,
    KernelsAndCommunications,
}

/// The time to issue a pass, and to have every member finish it.
fn pass(group: &ClientGroup, states: &[Handle], observed: Observed) -> (Duration, Duration) {
    let keeps = match observed {
        Observed::Not => None,
        Observed::Kernels => Some(TimingRequest::Deferred(Timed::Kernels)),
        Observed::KernelsAndCommunications => {
            Some(TimingRequest::Deferred(Timed::KernelsAndCommunications))
        }
    }
    .map(|timing| {
        Arc::new(Keeps {
            timing,
            unread: Mutex::default(),
        })
    });
    let watching = keeps.clone().map(|keeps| ExecutionObservation::new(keeps));

    let started = Instant::now();
    for _ in 0..STEPS {
        for (client, state) in group.clients().iter().zip(states) {
            unsafe {
                layer::launch_unchecked(
                    client,
                    CubeCount::Static((HIDDEN as u32).div_ceil(256), 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(state.clone(), HIDDEN),
                );
            }
        }
        let buffers = states
            .iter()
            .map(|state| MemberBuffers {
                src: state.clone(),
                dst: state.clone(),
            })
            .collect();
        group.all_reduce(buffers, F32, ReduceOperation::Mean);
    }
    let issued = started.elapsed();
    for client in group.clients() {
        block_on(client.sync()).expect("the member syncs");
    }
    let finished = started.elapsed();
    drop(watching);

    if let Some(keeps) = keeps {
        for measurement in std::mem::take(&mut *keeps.unread.lock().unwrap()) {
            block_on(measurement.resolve());
        }
    }
    (issued, finished)
}

fn main() {
    let group = ClientGroup::new(multi_device::clients()).expect("the devices form a group");
    let states: Vec<Handle> = group
        .clients()
        .iter()
        .map(|client| client.create_from_slice(f32::as_bytes(&vec![0.5; HIDDEN])))
        .collect();

    let order = [
        Observed::Not,
        Observed::Kernels,
        Observed::KernelsAndCommunications,
    ];
    for observed in order {
        pass(&group, &states, observed);
    }

    let mut times: Vec<Vec<(Duration, Duration)>> = vec![Vec::new(); order.len()];
    for rep in 0..REPS {
        // Reversed every other rep, so a drift does not favour one.
        let indices: Vec<usize> = match rep % 2 {
            0 => (0..order.len()).collect(),
            _ => (0..order.len()).rev().collect(),
        };
        for index in indices {
            times[index].push(pass(&group, &states, order[index]));
        }
    }

    println!(
        "{STEPS} steps of a kernel per member then an all-reduce of [1, {HIDDEN}] f32, {REPS} reps"
    );
    for (observed, times) in order.iter().zip(&times) {
        let per_step = |pick: fn(&(Duration, Duration)) -> Duration| {
            let mut each: Vec<f64> = times
                .iter()
                .map(|time| pick(time).as_secs_f64() * 1e6 / STEPS as f64)
                .collect();
            each.sort_by(f64::total_cmp);
            (each[each.len() / 2], each[0], each[each.len() - 1])
        };
        let (issue, issue_min, issue_max) = per_step(|time| time.0);
        let (step, step_min, step_max) = per_step(|time| time.1);
        println!(
            "{:>24}: issue {issue:7.2} us/step [{issue_min:.2}, {issue_max:.2}], \
             step {step:7.2} us/step [{step_min:.2}, {step_max:.2}]",
            format!("{observed:?}")
        );
    }
}

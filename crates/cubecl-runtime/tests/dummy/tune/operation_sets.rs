use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use cubecl_runtime::{
    server::Handle,
    tune::{AutotuneBound, Bounds, CloneInputGenerator, ResourceBound, Tunable, TunableSet},
};

use crate::dummy::{
    DummyClient, DummyElementwiseAddition, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, KernelTask, OneKernelAutotuneOperation,
};

use super::{DummyElementwiseAdditionBrokenCompilation, DummyElementwiseAdditionSlowWrong};

type TestSet = TunableSet<String, Vec<Handle>, ()>;

pub fn addition_set(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
) -> TunableSet<String, Vec<Handle>, ()> {
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());
    let op_add_slow = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    );
    TestSet::new(
        move |_input: &Vec<Handle>| format!("{}-{}", "add", log_shape_input_key(&shapes)),
        CloneInputGenerator,
    )
    .with(Tunable::new("add", move |inputs| op_add.run(inputs)))
    .with(Tunable::new("add_slow_wrong", move |inputs| {
        op_add_slow.run(inputs)
    }))
}

pub fn multiplication_set(client: DummyClient, shapes: Vec<Vec<usize>>) -> TestSet {
    let op_mul_slow = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplicationSlowWrong),
        client.clone(),
    );
    let op_mul = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplication),
        client.clone(),
    );
    TestSet::new(
        move |_input: &Vec<Handle>| format!("{}-{}", "mul", log_shape_input_key(&shapes)),
        CloneInputGenerator,
    )
    .with(Tunable::new("mul_slow_wrong", move |inputs| {
        op_mul_slow.run(inputs)
    }))
    .with(Tunable::new("mul", move |inputs| op_mul.run(inputs)))
}

/// Addition set with the slow+wrong kernel registered *first* and the fast+correct one
/// second, plus a single throughput [`AutotuneBound`]. Used to exercise the native
/// autotune short-circuit: the resulting `time_limit` is `(1 / throughput) / threshold`
/// seconds. A generous limit makes the tuner accept the first (slow) candidate and never
/// benchmark the faster one; an unreachable limit forces every candidate to be benchmarked.
pub fn bounded_addition_set_slow_first(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    throughput: f64,
    threshold: f32,
) -> TestSet {
    let op_add_slow = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    );
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());

    TestSet::new(
        move |_input: &Vec<Handle>| format!("{}-{}", "add_bounded", log_shape_input_key(&shapes)),
        CloneInputGenerator,
    )
    .with(Tunable::new("add_slow_wrong", move |inputs| {
        op_add_slow.run(inputs)
    }))
    .with(Tunable::new("add", move |inputs| op_add.run(inputs)))
    .with_bounds(Arc::new(move |_key: &String, _inputs: &Vec<Handle>| {
        Bounds {
            bounds: vec![AutotuneBound {
                resource: ResourceBound {
                    amount: 1,
                    peak_per_s: throughput,
                },
                threshold,
            }],
            launch_overhead: Duration::ZERO,
        }
    }))
}

pub fn log_shape_input_key(shapes: &[Vec<usize>]) -> String {
    let mut hash = String::new();
    let lhs = &shapes[0];
    for size in lhs {
        let exp = f32::ceil(f32::log2(*size as f32)) as u32;
        hash.push_str(2_u32.pow(exp).to_string().as_str());
        hash.push(',');
    }
    hash
}

/// Same generous bound as [`bounded_addition_set_slow_first`] but with
/// `with_short_circuit(false)`, so the tuner must benchmark every candidate
/// even though the first one qualifies.
pub fn bounded_addition_set_no_short_circuit(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
) -> TestSet {
    let op_add_slow = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    );
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());

    TestSet::new(
        move |_input: &Vec<Handle>| {
            format!("{}-{}", "add_bounded_nosc", log_shape_input_key(&shapes))
        },
        CloneInputGenerator,
    )
    .with(Tunable::new("add_slow_wrong", move |inputs| {
        op_add_slow.run(inputs)
    }))
    .with(Tunable::new("add", move |inputs| op_add.run(inputs)))
    .with_bounds(Arc::new(move |_key: &String, _inputs: &Vec<Handle>| {
        Bounds {
            bounds: vec![AutotuneBound {
                resource: ResourceBound {
                    amount: 1,
                    peak_per_s: 1.0,
                },
                threshold: 1.0,
            }],
            launch_overhead: Duration::ZERO,
        }
    }))
    .with_short_circuit(false)
}

/// Addition set whose first candidate always rejects its own configuration, standing in for a
/// kernel a backend refuses before compilation. `calls` counts how often the rejecting closure
/// runs, so a test can assert the benchmark gives up on the first failure.
///
/// `uid` is mixed into the autotune key so every run is a cache miss: the persistent cache
/// would otherwise answer from a previous run and no candidate would be benchmarked at all.
pub fn addition_set_with_rejected_candidate(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    uid: String,
    calls: Arc<AtomicUsize>,
) -> TestSet {
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());

    TestSet::new(
        move |_input: &Vec<Handle>| format!("add_rejected-{uid}-{}", log_shape_input_key(&shapes)),
        CloneInputGenerator,
    )
    .with(Tunable::new("add_rejected", move |_inputs| {
        calls.fetch_add(1, Ordering::Relaxed);
        Result::<(), String>::Err("unsupported by this device".to_string())
    }))
    .with(Tunable::new("add", move |inputs| op_add.run(inputs)))
}

/// Addition set with one candidate that sleeps per element, far enough behind the other two to be
/// eliminated on the numbers. Three candidates because the survivor floor keeps two alive no
/// matter what, so a two-candidate set can never eliminate anything.
///
/// Each closure counts its own calls, which is how a test sees that sampling actually stopped for
/// the slow one rather than merely that the fast one won. `uid` keeps the key a cache miss, as in
/// [`addition_set_with_rejected_candidate`].
pub fn addition_set_with_slow_candidate(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    uid: String,
    fast_calls: Arc<AtomicUsize>,
    slow_calls: Arc<AtomicUsize>,
) -> TestSet {
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());
    let op_add_other =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());
    let op_add_slow = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    );

    TestSet::new(
        move |_input: &Vec<Handle>| format!("add_slow-{uid}-{}", log_shape_input_key(&shapes)),
        CloneInputGenerator,
    )
    .with(Tunable::new("add", move |inputs| {
        fast_calls.fetch_add(1, Ordering::Relaxed);
        op_add.run(inputs)
    }))
    .with(Tunable::new("add_other", move |inputs| {
        op_add_other.run(inputs)
    }))
    .with(Tunable::new("add_slow_wrong", move |inputs| {
        slow_calls.fetch_add(1, Ordering::Relaxed);
        op_add_slow.run(inputs)
    }))
}

/// Addition set whose last candidate uses a kernel that fails to compile. The server
/// records the failure and returns it lazily at `end_profile`, which is how a real
/// backend reports a kernel it cannot compile: the tuner must skip the candidate on
/// that returned error, with no panic involved, and the surviving `add` wins.
///
/// The broken candidate goes last on purpose. Ahead of a working one, its errors get
/// swallowed by the next candidate's warmup flush, and the test would pass without
/// telling us whether anything stayed pending on the process-global dummy server.
///
/// `uid` keeps the key a cache miss, as in [`addition_set_with_rejected_candidate`].
pub fn addition_set_with_failing_compilation(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    uid: String,
) -> TestSet {
    let op_broken = OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionBrokenCompilation),
        client.clone(),
    );
    let op_add =
        OneKernelAutotuneOperation::new(KernelTask::new(DummyElementwiseAddition), client.clone());

    TestSet::new(
        move |_input: &Vec<Handle>| {
            format!("add_no_compile-{uid}-{}", log_shape_input_key(&shapes))
        },
        CloneInputGenerator,
    )
    .with(Tunable::new("add", move |inputs| op_add.run(inputs)))
    .with(Tunable::new("add_no_compile", move |inputs| {
        op_broken.run(inputs)
    }))
}

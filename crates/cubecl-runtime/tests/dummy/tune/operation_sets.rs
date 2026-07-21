use std::{sync::Arc, time::Duration};

use cubecl_runtime::{
    server::Handle,
    tune::{AutotuneBound, Bounds, CloneInputGenerator, Tunable, TunableSet},
};

use crate::dummy::{
    DummyClient, DummyElementwiseAddition, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, KernelTask, OneKernelAutotuneOperation,
};

use super::DummyElementwiseAdditionSlowWrong;

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
                throughput,
                threshold,
                ops_count: 1,
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

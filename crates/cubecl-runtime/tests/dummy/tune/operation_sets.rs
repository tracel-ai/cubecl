use cubecl_runtime::{
    server::Binding,
    tune::{Tunable, TunableSet},
};

use crate::dummy::{
    DummyClient, DummyElementwiseAddition, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, KernelTask, OneKernelAutotuneOperation,
};

use super::DummyElementwiseAdditionSlowWrong;

#[allow(clippy::ptr_arg, reason = "Needed for type inference")]
fn clone_bindings(_key: &String, bindings: &Vec<Binding>) -> Vec<Binding> {
    bindings.clone()
}

type TestSet = TunableSet<String, Vec<Binding>, ()>;

pub fn addition_set(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
) -> TunableSet<String, Vec<Binding>, ()> {
    TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "add", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with(Tunable::new(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAddition),
        client.clone(),
    )))
    .with(Tunable::new(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    )))
}

pub fn multiplication_set(client: DummyClient, shapes: Vec<Vec<usize>>) -> TestSet {
    TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "mul", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with(Tunable::new(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplicationSlowWrong),
        client.clone(),
    )))
    .with(Tunable::new(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplication),
        client.clone(),
    )))
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

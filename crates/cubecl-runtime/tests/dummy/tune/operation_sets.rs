#[cfg(std_io)]
use rand::{Rng, distr::Alphanumeric};

use cubecl_runtime::{
    server::{Binding, Bindings, CubeCount},
    tune::{AsFunctionTunable, TunableSet},
};

use crate::{
    DummyKernel,
    dummy::{
        CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
        DummyElementwiseMultiplication, DummyElementwiseMultiplicationSlowWrong, KernelTask,
        OneKernelAutotuneOperation,
    },
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
    .with_tunable(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAddition),
        client.clone(),
    ))
    .with_tunable(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    ))
}

pub fn multiplication_set(client: DummyClient, shapes: Vec<Vec<usize>>) -> TestSet {
    TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "mul", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with_tunable(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplicationSlowWrong),
        client.clone(),
    ))
    .with_tunable(OneKernelAutotuneOperation::new(
        KernelTask::new(DummyElementwiseMultiplication),
        client.clone(),
    ))
}

pub fn cache_test_set(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    bindings: Vec<Binding>,
    generate_random_checksum: bool,
) -> TestSet {
    fn tunable(
        client: DummyClient,
        kernel: impl DummyKernel,
        bindings: Vec<Binding>,
    ) -> impl Fn(Vec<Binding>) {
        let kernel = KernelTask::new(kernel);
        move |_| {
            client.execute(
                kernel.clone(),
                CubeCount::Static(1, 1, 1),
                Bindings::new().with_buffers(bindings.clone()),
            );
        }
    }
    let mut set = TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "cache_test", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with_tunable(tunable(client.clone(), CacheTestFastOn3, bindings.clone()).ok())
    .with_tunable(tunable(client.clone(), CacheTestSlowOn3, bindings.clone()).ok());
    if generate_random_checksum {
        set = set.with_custom_checksum(|_| {
            let rand_string: String = rand::rng()
                .sample_iter(&Alphanumeric)
                .take(16)
                .map(char::from)
                .collect();
            rand_string
        });
    }
    set
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

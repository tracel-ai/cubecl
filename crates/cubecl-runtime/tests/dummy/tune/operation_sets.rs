#[cfg(autotune_persistent_cache)]
use rand::{distributions::Alphanumeric, Rng};
use std::sync::Arc;

use cubecl_runtime::{server::Binding, tune::TunableSet};

use crate::dummy::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationSlowWrong,
    OneKernelAutotuneOperation,
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
        Arc::new(DummyElementwiseAddition),
        client.clone(),
    ))
    .with_tunable(OneKernelAutotuneOperation::new(
        Arc::new(DummyElementwiseAdditionSlowWrong),
        client.clone(),
    ))
}

pub fn multiplication_set(client: DummyClient, shapes: Vec<Vec<usize>>) -> TestSet {
    TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "mul", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with_tunable(OneKernelAutotuneOperation::new(
        Arc::new(DummyElementwiseMultiplicationSlowWrong),
        client.clone(),
    ))
    .with_tunable(OneKernelAutotuneOperation::new(
        Arc::new(DummyElementwiseMultiplication),
        client.clone(),
    ))
}

pub fn cache_test_set(
    client: DummyClient,
    shapes: Vec<Vec<usize>>,
    generate_random_checksum: bool,
) -> TestSet {
    let mut set = TestSet::new(
        move |_input: &Vec<Binding>| format!("{}-{}", "cache_test", log_shape_input_key(&shapes)),
        clone_bindings,
    )
    .with_tunable(OneKernelAutotuneOperation::new(
        Arc::new(CacheTestFastOn3),
        client.clone(),
    ))
    .with_tunable(OneKernelAutotuneOperation::new(
        Arc::new(CacheTestSlowOn3),
        client.clone(),
    ));
    if generate_random_checksum {
        set = set.with_custom_checksum(|_| {
            let rand_string: String = rand::thread_rng()
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

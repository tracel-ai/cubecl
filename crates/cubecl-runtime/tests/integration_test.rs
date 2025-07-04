mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, test_client};

use cubecl_runtime::server::Bindings;
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::{local_tuner, tune::LocalTuner};
use dummy::*;

#[test]
fn created_resource_is_the_same_when_read() {
    let client = test_client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create(&resource);

    let obtained_resource = client.read_one(resource_description.binding());

    assert_eq!(resource, obtained_resource)
}

#[test]
fn empty_allocates_memory() {
    let client = test_client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read_one(resource_description.binding());

    assert_eq!(empty_resource.len(), 4);
}

#[test]
fn execute_elementwise_addition() {
    let client = test_client(&DummyDevice);
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);

    client.execute(
        KernelTask::new(DummyElementwiseAddition),
        CubeCount::Static(1, 1, 1),
        Bindings::new().with_buffers(vec![lhs.binding(), rhs.binding(), out.clone().binding()]),
    );

    let obtained_resource = client.read_one(out.binding());

    assert_eq!(obtained_resource, Vec::from([4, 5, 6]))
}

#[test]
#[cfg(feature = "std")]
fn autotune_basic_addition_execution() {
    static TUNER: LocalTuner<String, String> = local_tuner!("autotune_basic_addition_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs.binding(), rhs.binding(), out.clone().binding()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::addition_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out.binding());

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([4, 5, 6]));
}

#[test]
#[cfg(feature = "std")]
fn autotune_basic_multiplication_execution() {
    static TUNER: LocalTuner<String, String> =
        local_tuner!("autotune_basic_multiplication_execution");

    let client = test_client(&DummyDevice);

    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs.binding(), rhs.binding(), out.clone().binding()];

    let test_set = TUNER.init(|| {
        let client = test_client(&DummyDevice);
        let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
        dummy::multiplication_set(client, shapes)
    });
    TUNER.execute(&"test".to_string(), &client, test_set, handles);

    let obtained_resource = client.read_one(out.binding());

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([0, 4, 8]));
}

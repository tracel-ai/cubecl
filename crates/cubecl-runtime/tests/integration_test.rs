mod dummy;

use crate::dummy::{DummyDevice, DummyElementwiseAddition, client};

use cubecl_runtime::server::CubeCount;
use cubecl_runtime::{ComputeRuntime, server::Bindings};
use dummy::*;

#[allow(unused)]
use serial_test::serial;

type Runtime = ComputeRuntime<DummyDevice, dummy::DummyServer, dummy::DummyChannel>;

#[test]
fn created_resource_is_the_same_when_read() {
    let client = client(&DummyDevice);
    let resource = Vec::from([0, 1, 2]);
    let resource_description = client.create(&resource);

    let obtained_resource = client.read_one(resource_description.binding());

    assert_eq!(resource, obtained_resource)
}

#[test]
fn empty_allocates_memory() {
    let client = client(&DummyDevice);
    let size = 4;
    let resource_description = client.empty(size);
    let empty_resource = client.read_one(resource_description.binding());

    assert_eq!(empty_resource.len(), 4);
}

#[test]
fn execute_elementwise_addition() {
    let client = client(&DummyDevice);
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
#[serial]
#[cfg(feature = "std")]
fn autotune_basic_addition_execution() {
    TEST_TUNER.clear();
    let client = client(&DummyDevice);

    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs.binding(), rhs.binding(), out.clone().binding()];

    let test_set = dummy::addition_set(client.clone(), shapes);
    autotune_execute(&client, &test_set, handles);

    let obtained_resource = client.read_one(out.binding());

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([4, 5, 6]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_basic_multiplication_execution() {
    TEST_TUNER.clear();
    let client = client(&DummyDevice);

    let shapes = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs = client.create(&[0, 1, 2]);
    let rhs = client.create(&[4, 4, 4]);
    let out = client.empty(3);
    let handles = vec![lhs.binding(), rhs.binding(), out.clone().binding()];

    let test_set = dummy::multiplication_set(client.clone(), shapes);
    autotune_execute(&client, &test_set, handles);

    let obtained_resource = client.read_one(out.binding());

    // If slow kernel was selected it would output [0, 1, 2]
    assert_eq!(obtained_resource, Vec::from([0, 4, 8]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_cache_same_key_return_a_cache_hit() {
    TEST_TUNER.clear();
    let runtime = Runtime::new();

    let client = runtime.client(&DummyDevice, dummy::init_client);

    // note: the key name depends on the shapes of the operation set
    // see log_shape_input_key for more info.

    // in this test both shapes [1,3] and [1,4] end up with the same key name
    // which is 'cache_test-1,4'
    let shapes_1 = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs_1 = client.create(&[0, 1, 2]);
    let rhs_1 = client.create(&[4, 4, 4]);
    let out_1 = client.empty(3);
    let handles_1 = vec![lhs_1.binding(), rhs_1.binding(), out_1.binding()];

    let shapes_2 = vec![vec![1, 4], vec![1, 4], vec![1, 4]];
    let lhs_2 = client.create(&[0, 1, 2, 3]);
    let rhs_2 = client.create(&[5, 6, 7, 8]);
    let out_2 = client.empty(4);
    let handles_2 = vec![lhs_2.binding(), rhs_2.binding(), out_2.clone().binding()];

    let cache_test_autotune_kernel_1 =
        dummy::cache_test_set(client.clone(), shapes_1, handles_1, false);
    let cache_test_autotune_kernel_2 =
        dummy::cache_test_set(client.clone(), shapes_2, handles_2, false);
    autotune_execute(&client, &cache_test_autotune_kernel_1, vec![]);
    autotune_execute(&client, &cache_test_autotune_kernel_2, vec![]);

    let obtained_resource = client.read_one(out_2.binding());

    // Cache should be hit, so CacheTestFastOn3 should be used, returning lhs
    assert_eq!(obtained_resource, Vec::from([0, 1, 2, 3]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_cache_different_keys_return_a_cache_miss() {
    TEST_TUNER.clear();
    let client = client(&DummyDevice);

    // in this test shapes [1,3] and [1,5] ends up with different key names
    // which are 'cache_test-1,4' and 'cache_test-1,8'
    let shapes_1 = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs_1 = client.create(&[0, 1, 2]);
    let rhs_1 = client.create(&[4, 4, 4]);
    let out_1 = client.empty(3);
    let handles_1 = vec![lhs_1.binding(), rhs_1.binding(), out_1.binding()];

    let shapes_2 = vec![vec![1, 5], vec![1, 5], vec![1, 5]];
    let lhs_2 = client.create(&[0, 1, 2, 3, 4]);
    let rhs_2 = client.create(&[5, 6, 7, 8, 9]);
    let out_2 = client.empty(5);
    let handles_2 = vec![lhs_2.binding(), rhs_2.binding(), out_2.clone().binding()];

    let cache_test_autotune_kernel_1 =
        dummy::cache_test_set(client.clone(), shapes_1, handles_1, false);
    let cache_test_autotune_kernel_2 =
        dummy::cache_test_set(client.clone(), shapes_2, handles_2, false);
    autotune_execute(&client, &cache_test_autotune_kernel_1, vec![]);
    autotune_execute(&client, &cache_test_autotune_kernel_2, vec![]);

    let obtained_resource = client.read_one(out_2.binding());

    // Cache should be missed, so CacheTestSlowOn3 (but faster on 5) should be used, returning rhs
    assert_eq!(obtained_resource, Vec::from([5, 6, 7, 8, 9]));
}

#[test]
#[serial]
#[cfg(feature = "std")]
fn autotune_cache_different_checksums_return_a_cache_miss() {
    TEST_TUNER.clear();

    let runtime = Runtime::new();
    let client = runtime.client(&DummyDevice, dummy::init_client);

    // in this test both shapes [1,3] and [1,4] end up with the same key name
    // which is 'cache_test-1,4'
    let shapes_1 = vec![vec![1, 3], vec![1, 3], vec![1, 3]];
    let lhs_1 = client.create(&[0, 1, 2]);
    let rhs_1 = client.create(&[4, 4, 4]);
    let out_1 = client.empty(3);
    let handles_1 = vec![lhs_1.binding(), rhs_1.binding(), out_1.binding()];
    let cache_test_autotune_kernel_1 =
        dummy::cache_test_set(client.clone(), shapes_1, handles_1, false);
    autotune_execute(&client, &cache_test_autotune_kernel_1, vec![]);

    TEST_TUNER.clear();

    let shapes_2 = vec![vec![1, 4], vec![1, 4], vec![1, 4]];
    let lhs_2 = client.create(&[0, 1, 2, 3]);
    let rhs_2 = client.create(&[5, 6, 7, 8]);
    let out_2 = client.empty(4);
    let handles_2 = vec![lhs_2.binding(), rhs_2.binding(), out_2.clone().binding()];

    let cache_test_autotune_kernel_2 =
        dummy::cache_test_set(client.clone(), shapes_2, handles_2, true);
    autotune_execute(&client, &cache_test_autotune_kernel_2, vec![]);

    let obtained_resource = client.read_one(out_2.binding());

    // Cache should be missed because the checksum on 4 is generated randomly
    // and thus is always different,
    // so CacheTestSlowOn3 (but faster on 4) should be used, returning rhs
    assert_eq!(obtained_resource, Vec::from([5, 6, 7, 8]));
}

#[test]
#[should_panic]
fn test_recursive_execution_same_state() {
    let device_id = DeviceId {
        type_id: 0,
        index_id: 5,
    };
    let context = DeviceHandle::<TestDeviceState<1>>::new(device_id);
    let context_cloned = context.clone();

    let _count = context
        .submit_blocking(move |state| {
            state.counter += 1;
            context_cloned.submit(move |state| {
                state.counter += 1;
            });
        })
        .unwrap();
}

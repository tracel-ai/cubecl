#[test]
#[should_panic]
#[cfg(not(miri))]
#[allow(clippy::all)]
fn test_recursive_execution_same_state() {
    let device_id = DeviceId {
        type_id: 10,
        index_id: 5,
    };
    let handle = DeviceHandle::<TestDeviceState<1>>::new(device_id);
    let handle_cloned = handle.clone();

    let _count = handle
        .submit_blocking(move |state| {
            state.counter += 1;
            handle_cloned.submit(move |state| {
                state.counter += 1;
            })
        })
        .unwrap();
    handle.submit_blocking(|_state| {}).unwrap();
}

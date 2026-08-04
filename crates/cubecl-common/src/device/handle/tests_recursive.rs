#[test]
#[should_panic]
#[cfg(not(miri))]
#[allow(clippy::all)]
fn test_recursive_execution_same_state() {
    let handle = DeviceFixture::new(
        DeviceHandle::<TestDeviceState<1>>::new,
        DeviceHandle::<TestDeviceState<1>>::shutdown,
    );
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

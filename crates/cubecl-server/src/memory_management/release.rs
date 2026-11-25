/// Release `memory` after a wait failed with `err`, only when the device is
/// poisoned and can no longer use it. Otherwise completion is unknown, so the
/// allocations stay reserved: their owners are leaked.
pub(crate) fn release_or_leak<T>(memory: T, err: &crate::server::ServerError) {
    if err.is_device_poisoned() {
        return core::mem::drop(memory);
    }
    core::mem::forget(memory);
    log::error!("leaking memory the device may still be using, after a failed wait: {err}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerError;
    use alloc::rc::Rc;

    /// Memory goes only once the device is poisoned; after any other failed
    /// wait it is leaked.
    #[test]
    fn memory_is_released_only_on_a_poisoned_device() {
        let poisoned: ServerError = crate::driver::DevicePoison::new("status 700").into();
        let other = ServerError::Generic {
            reason: "the wait failed".into(),
            backtrace: Default::default(),
        };
        let memory = Rc::new(());

        release_or_leak(memory.clone(), &poisoned);
        assert_eq!(Rc::strong_count(&memory), 1, "released");
        release_or_leak(memory.clone(), &other);
        assert_eq!(Rc::strong_count(&memory), 2, "leaked");
    }
}

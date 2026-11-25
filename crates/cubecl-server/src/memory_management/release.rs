/// Release memory after a failed wait only when the device is poisoned and can no longer use it.
/// Otherwise completion is unknown, so keep the allocations reserved by leaking their owners.
pub(crate) fn release_or_leak<T>(memory: T, err: &crate::server::ServerError) {
    if err.is_device_poisoned() {
        core::mem::drop(memory);
    } else {
        core::mem::forget(memory);
        log::error!("leaking memory the device may still be using, after a failed wait: {err}");
    }
}

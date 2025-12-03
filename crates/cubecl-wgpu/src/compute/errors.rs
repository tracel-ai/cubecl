use cubecl_common::stub::Mutex;
use cubecl_core::future::DynFut;

static ERROR_SCOPES_LOCK: Mutex<u32> = Mutex::new(0);

/// Fetches lazy errors on the device.
///
/// # Notes
///
/// Each call to this function should be done after a call to [watch_error].
pub(crate) fn fetch_error(device: &wgpu::Device) -> DynFut<Option<wgpu::Error>> {
    let mut error_scope = ERROR_SCOPES_LOCK.lock().unwrap();

    if *error_scope > 0 {
        let error = device.pop_error_scope();
        *error_scope -= 1;
        core::mem::drop(error_scope);

        return Box::pin(error);
    } else {
        core::mem::drop(error_scope);
    };

    Box::pin(async move { None })
}

/// Tracks lazy errors on a device.
pub(crate) fn track_error(device: &wgpu::Device, filter: wgpu::ErrorFilter) {
    let mut error_scope = ERROR_SCOPES_LOCK.lock().unwrap();
    *error_scope += 1;
    device.push_error_scope(filter);
    core::mem::drop(error_scope);
}

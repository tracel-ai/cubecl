use crate::device::{DeviceId, DeviceService};

/// An error happened while executing a call.
#[derive(Debug)]
pub struct CallError;

#[derive(new, Clone, Debug)]
/// Error when creating a [`DeviceService`].
pub struct ServiceCreationError {
    #[allow(dead_code)] // Debug uses it.
    reason: alloc::string::String,
}

pub(crate) trait DeviceHandleSpec<S: DeviceService>: Sized {
    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError>;

    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn new(device_id: DeviceId) -> Self;

    /// Executes a task on the dedicated device thread and returns the result of the task.
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::submit`] if you don't need to wait for a returned type.
    fn submit_blocking<R: Send + 'static, T: FnOnce(&mut S) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError>;

    /// Executes a task on the dedicated device thread and returns the result of the task.
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::submit_blocking`] if you don't need to have scope execution garantee,
    /// which requires an extra allocation.
    fn submit_blocking_scoped<'a, R: Send + 'a, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> R;

    /// Submit a task for execution on the dedicated device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T);

    /// TODO: Docs.
    fn exclusive<R: Send + 'static, T: FnOnce() -> R + Send + 'static>(
        &self,
        task: T,
    ) -> Result<R, CallError>;

    /// TODO: Docs.
    fn exclusive_scoped<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError>;
}

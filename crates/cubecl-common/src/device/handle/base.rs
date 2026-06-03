use crate::device::{DeviceId, DeviceService, ServerUtilitiesHandle};
use alloc::boxed::Box;
use alloc::string::String;
use core::any::Any;

/// An error returned by a blocking device call.
///
/// When a task panics on a device's runner thread, the panic payload is
/// captured and carried here instead of being logged and dropped.
/// A `CallError` without a payload means the runner channel disconnected
/// before the task could produce a result.
pub struct CallError {
    /// The captured panic payload, or `None` if the call failed because the
    /// runner channel disconnected rather than because the task panicked.
    payload: Option<Box<dyn Any + Send>>,
}

impl CallError {
    // Only the `channel` backend (active when `multi_threading` is set) calls these
    // constructors. In `no_std` builds `mod channel` isn't compiled, and on wasm+std
    // the channel handle isn't the selected `Inner`, so the callers are dead — allow
    // the dead-code lint there while keeping it active in the multi-threaded build.
    /// Builds an error from a panic payload captured by `catch_unwind`.
    #[cfg_attr(not(multi_threading), allow(dead_code))]
    pub(crate) fn from_panic(payload: Box<dyn Any + Send>) -> Self {
        Self {
            payload: Some(payload),
        }
    }

    /// Builds an error for a runner channel that disconnected before producing
    /// a result (no panic payload available).
    #[cfg_attr(not(multi_threading), allow(dead_code))]
    pub(crate) fn disconnected() -> Self {
        Self { payload: None }
    }

    /// Returns the panic message if the failure was a panic whose payload is a
    /// string (the common case for `panic!`, indexing, `unwrap`, ...).
    pub fn message(&self) -> Option<&str> {
        let payload = self.payload.as_ref()?;
        if let Some(s) = payload.downcast_ref::<&'static str>() {
            Some(*s)
        } else if let Some(s) = payload.downcast_ref::<String>() {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Consumes the error and returns the captured panic payload, if any.
    ///
    /// The payload can be handed to [`std::panic::resume_unwind`] to re-raise
    /// the original panic on the caller's thread.
    pub fn into_panic(self) -> Option<Box<dyn Any + Send>> {
        self.payload
    }
}

impl core::fmt::Debug for CallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.message() {
            Some(message) => write!(
                f,
                "CallError(task panicked on device runner thread: {message})"
            ),
            None if self.payload.is_some() => f.write_str(
                "CallError(task panicked on device runner thread with a non-string payload)",
            ),
            None => f.write_str(
                "CallError(device runner channel disconnected before producing a result)",
            ),
        }
    }
}

#[derive(new, Clone, Debug)]
/// Error when creating a [`DeviceService`].
pub struct ServiceCreationError {
    #[allow(dead_code)] // Debug uses it.
    reason: alloc::string::String,
}

pub(crate) trait DeviceHandleSpec<S: DeviceService>: Sized {
    /// If functions block the current thread even if they are non-blocking.
    const BLOCKING: bool;

    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn insert(device_id: DeviceId, service: S) -> Result<Self, ServiceCreationError>;

    /// Creates or retrieves a context for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn new(device_id: DeviceId) -> Self;

    /// Retrieves the device ID for this handle.
    fn device_id(&self) -> DeviceId;

    /// Retrieves the server utilities for this thread.
    fn utilities(&self) -> ServerUtilitiesHandle;

    /// Doesn't flush the service state, but flushes any task enqueued in the communication
    /// channel.
    ///
    /// # Notes
    ///
    /// This is often not necessary, except for distributed operations.
    fn flush_queue(&self);

    /// Executes a task on the dedicated device thread and returns the result of the task.
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::submit`] if you don't need to wait for a returned type.
    fn submit_blocking<'a, R: Send, T: FnOnce(&mut S) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError>;

    /// Submit a task for execution on the dedicated device thread.
    fn submit<T: FnOnce(&mut S) + Send + 'static>(&self, task: T);

    /// TODO: Docs.
    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError>;
}

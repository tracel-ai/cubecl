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
    /// runner channel disconnected
    payload: Option<Box<dyn Any + Send>>,
}

impl CallError {
    // Only the `channel` backend (active when `multi_threading` is set) calls these
    // constructors.
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

    /// Re-raises the failure on the current thread, diverging.
    #[track_caller]
    pub fn resume(self) -> ! {
        match self.into_panic() {
            #[cfg(feature = "std")]
            Some(payload) => std::panic::resume_unwind(payload),
            // std::panic::resume_unwind doesn't exist in a no_std build
            #[cfg(not(feature = "std"))]
            Some(_payload) => {
                panic!("a device task panicked but its payload cannot be re-raised without `std`")
            }
            None => panic!("device runner channel disconnected before producing a result"),
        }
    }
}

/// Extension trait for re-raising a [`CallError`] on the caller's thread.
pub trait CallResultExt<R> {
    /// Returns the success value, or re-raises the original panic captured in the
    /// [`CallError`] via [`CallError::resume`].
    ///
    /// Use this instead of [`Result::unwrap`] to panics with the original payload itself.
    fn unwrap_or_resume(self) -> R;
}

impl<R> CallResultExt<R> for Result<R, CallError> {
    #[track_caller]
    fn unwrap_or_resume(self) -> R {
        match self {
            Ok(value) => value,
            Err(err) => err.resume(),
        }
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

/// What a caught panic payload says, where it says anything at all.
#[cfg(feature = "std")]
pub(crate) fn panic_reason(payload: &alloc::boxed::Box<dyn core::any::Any + Send>) -> &str {
    if let Some(reason) = payload.downcast_ref::<&'static str>() {
        return reason;
    }

    match payload.downcast_ref::<alloc::string::String>() {
        Some(reason) => reason.as_str(),
        None => "the service panicked with a payload that is not a string",
    }
}

#[derive(new, Clone, Debug)]
/// Error when creating a [`DeviceService`].
pub struct ServiceCreationError {
    reason: alloc::string::String,
}

impl ServiceCreationError {
    /// What the service said on the way down.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl core::fmt::Display for ServiceCreationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl core::error::Error for ServiceCreationError {}

/// How a service's state is reached: the transport under a [`DeviceHandle`].
///
/// A transport holds no service type. It hands a task the boxed state as
/// `&mut dyn Any`; the handle above it knows what that is.
pub trait DeviceHandleSpec: Sized + Clone {
    /// If functions block the current thread even if they are non-blocking.
    const BLOCKING: bool;

    /// Registers `service` for the device and returns a transport to it.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    fn insert<S: DeviceService>(
        device_id: DeviceId,
        service: S,
    ) -> Result<Self, ServiceCreationError>;

    /// Creates or retrieves a transport for the given device ID.
    ///
    /// If a runner thread for this `device_id` does not exist, it will be spawned.
    ///
    /// The error is the service failing to start — a driver that is missing,
    /// a device that will not open. A transport with no way to catch a panic
    /// crossing back to the caller lets it through instead, so a caller that
    /// must not die still has to be ready for one.
    fn try_new<S: DeviceService>(device_id: DeviceId) -> Result<Self, ServiceCreationError>;

    /// [`try_new`](Self::try_new), for the caller with nothing to do about a
    /// service that will not start.
    ///
    /// # Panics
    ///
    /// Where the service fails to start.
    fn new<S: DeviceService>(device_id: DeviceId) -> Self {
        match Self::try_new::<S>(device_id) {
            Ok(handle) => handle,
            Err(err) => panic!("{err:?}"),
        }
    }

    /// Retrieves the device ID for this handle.
    fn device_id(&self) -> DeviceId;

    /// Retrieves the server utilities for this thread.
    fn utilities(&self) -> ServerUtilitiesHandle;

    /// Doesn't flush the service state, but flushes any task enqueued in the communication
    /// channel.
    ///
    /// # Notes
    ///
    /// This is a no-op for blocking handles.
    fn flush_queue(&self);

    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError>;

    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T);

    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError>;

    fn shutdown(device_id: DeviceId) {
        let _ = device_id;
    }
}

#[cfg(test)]
mod tests {
    use super::CallError;
    use alloc::boxed::Box;
    use core::any::Any;

    /// D.1 — A disconnected error carries neither a message nor a payload, which is
    /// what distinguishes it from a non-string panic (both have `message() == None`).
    #[test]
    fn test_disconnected_has_no_message_or_payload() {
        let err = CallError::disconnected();
        assert_eq!(err.message(), None);
        assert!(err.into_panic().is_none());
    }

    /// D.2 — Debug of a string payload includes the panic message.
    #[test]
    fn test_debug_string_payload_includes_message() {
        let err = CallError::from_panic(Box::new("oops") as Box<dyn Any + Send>);
        let debug = alloc::format!("{err:?}");
        assert!(debug.contains("task panicked on device runner thread"));
        assert!(debug.contains("oops"));
    }

    /// D.3 — Debug of a non-string payload reports it as such.
    #[test]
    fn test_debug_non_string_payload() {
        let err = CallError::from_panic(Box::new(123u8) as Box<dyn Any + Send>);
        let debug = alloc::format!("{err:?}");
        assert!(debug.contains("non-string payload"), "got: {debug}");
    }

    /// D.4 — Debug of a disconnected error mentions the disconnection.
    #[test]
    fn test_debug_disconnected() {
        let err = CallError::disconnected();
        let debug = alloc::format!("{err:?}");
        assert!(debug.contains("disconnected"), "got: {debug}");
    }
}

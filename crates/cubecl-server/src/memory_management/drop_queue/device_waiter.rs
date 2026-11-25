use crate::server::ServerError;

/// Waits until the device has finished processing every command submitted
/// before it was created: a fence.
pub trait DeviceWaiter: Sized {
    /// Block the current thread until the device signals this fence.
    ///
    /// # Guarantees
    ///
    /// `Ok` only once every command submitted before the fence completed.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream itself failed.
    fn wait(self) -> Result<(), ServerError>;
}

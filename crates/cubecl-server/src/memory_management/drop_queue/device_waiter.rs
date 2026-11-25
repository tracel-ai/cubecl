use crate::server::ServerError;

/// Waits until the stream it was recorded on has finished processing every
/// command submitted to it before it was created: a fence.
pub trait DeviceWaiter: Sized {
    /// Block the current thread until the device signals this fence.
    ///
    /// # Guarantees
    ///
    /// `Ok` only once every command submitted to the fence's stream before it
    /// was recorded completed. Other streams' work is not waited on.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream itself failed.
    fn wait(self) -> Result<(), ServerError>;
}

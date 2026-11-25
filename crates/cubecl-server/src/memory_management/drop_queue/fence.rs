use crate::server::ServerError;

/// A synchronization primitive that blocks until the device has finished
/// processing all commands submitted before the fence was created.
pub trait Fence: Sized {
    /// Block the current thread until the device signals this fence.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream itself failed.
    fn wait(self) -> Result<(), ServerError>;

    /// [`wait`](Self::wait).
    ///
    /// What the drop queue needs: it only has to know the device is done
    /// reading the memory it is about to free. A poisoned device is done either
    /// way, and the fault reaches the caller through whatever it touches next.
    /// Any other failure leaves that unknown, so the memory must not be freed.
    fn sync(self) -> bool {
        match self.wait() {
            Ok(()) => true,
            Err(err) if err.is_device_poisoned() => true,
            Err(err) => {
                log::error!("a fence wait failed, keeping the memory it guards: {err}");
                false
            }
        }
    }
}

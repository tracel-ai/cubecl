use crate::server::ServerError;

/// Waits until the device has finished processing every command submitted
/// before it was created: a fence.
pub trait DeviceWaiter: Sized {
    /// Block the current thread until the device signals this fence.
    ///
    /// # Errors
    ///
    /// The fault the wait reveals, when the stream itself failed.
    fn wait(self) -> Result<(), ServerError>;

    /// [`wait`](Self::wait), answering whether the device is done reading the
    /// memory the fence guards.
    ///
    /// What the drop queue needs: it only has to know the device is done
    /// reading the memory it is about to free. A poisoned device is done either
    /// way, and the fault reaches the caller through whatever it touches next.
    /// Any other failure leaves that unknown, so the memory must not be freed.
    fn sync(self) -> bool {
        self.wait().map_or_else(done_after, |()| true)
    }
}

/// Whether the device is done after a wait failed with `err`: only once it is
/// poisoned and runs nothing any more.
fn done_after(err: ServerError) -> bool {
    if err.is_device_poisoned() {
        return true;
    }
    log::error!("a fence wait failed, keeping the memory it guards: {err}");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fence whose wait answers `result`.
    struct Waited(Result<(), ServerError>);

    impl DeviceWaiter for Waited {
        fn wait(self) -> Result<(), ServerError> {
            self.0
        }
    }

    #[test]
    fn memory_is_freed_only_once_the_device_is_done_with_it() {
        let poisoned: ServerError = crate::driver::DevicePoison::new("status 700").into();
        let other = ServerError::Generic {
            reason: "the wait failed".into(),
            backtrace: Default::default(),
        };

        assert!(Waited(Ok(())).sync(), "a fence that was reached");
        assert!(
            Waited(Err(poisoned)).sync(),
            "a poisoned device runs nothing anymore"
        );
        assert!(
            !Waited(Err(other)).sync(),
            "after any other failure, the device may still be reading the memory"
        );
    }
}

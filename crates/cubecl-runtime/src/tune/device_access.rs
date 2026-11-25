use crate::client::{Client, ProfileWindow};
use crate::server::{ProfileError, ServerError};
#[cfg(feature = "std")]
use alloc::string::String;
use cubecl_common::profile::ProfileDuration;
#[cfg(feature = "std")]
use cubecl_environment::backtrace::BackTrace;

/// How a tune holds the device it measures while it runs its candidates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuneDeviceAccess {
    /// The device is held to itself for every batch and every sample, nothing else scheduled
    /// against it: the candidates run on the device's own thread.
    #[default]
    Exclusive,
    /// The device is never held: the candidates run on the calling thread, each sample timed by a
    /// window opened and closed on the calling stream. Work a candidate hands to another thread,
    /// such as a lazy queue drained there, reaches the device and lands on that stream; work of
    /// other streams may run beside a sample and slow it.
    Shared,
}

impl TuneDeviceAccess {
    /// What `task` returns, run with `client`'s device held where this access holds it.
    ///
    /// # Errors
    ///
    /// Where the device could not be taken exclusively.
    pub(crate) fn hold<'a, R: Send + 'static>(
        self,
        client: &'a Client,
        task: impl FnOnce() -> R + Send + 'a,
    ) -> Result<R, ServerError> {
        match self {
            Self::Exclusive => client.exclusive(task),
            Self::Shared => Ok(task()),
        }
    }

    /// What `task` returns, run as this access runs a candidate, its panic contained: in a device
    /// call of its own where the device is held, on the calling thread where it never is.
    ///
    /// # Errors
    ///
    /// Why `task` stopped: its panic, under a hold already taken ([`hold`](Self::hold)).
    pub(crate) fn contain<'a, R: Send + 'static>(
        self,
        client: &'a Client,
        task: impl FnOnce() -> R + Send + 'a,
    ) -> Result<R, ServerError> {
        match self {
            Self::Exclusive => client.exclusive(task),
            Self::Shared => caught(task),
        }
    }

    /// What `func` returns and how long its launches took on the calling stream of `client`.
    ///
    /// # Errors
    ///
    /// Why the window did not open or close.
    pub(crate) fn profile<O: Send + 'static>(
        self,
        client: &Client,
        func: impl FnOnce() -> O + Send,
        name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        match self {
            Self::Exclusive => client.profile(func, name),
            Self::Shared => profile_on_stream(client, func),
        }
    }
}

/// What `task` returns, or the panic it stopped on. Without `std` a panic is never caught.
fn caught<R>(task: impl FnOnce() -> R) -> Result<R, ServerError> {
    #[cfg(feature = "std")]
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).map_err(|payload| {
            ServerError::Generic {
                reason: payload
                    .downcast_ref::<&str>()
                    .map(|message| String::from(*message))
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| String::from("a panic with no message")),
                backtrace: BackTrace::capture(),
            }
        })
    }
    #[cfg(not(feature = "std"))]
    {
        Ok(task())
    }
}

/// `func` timed between a window opened and closed on `client`'s calling stream, the window
/// abandoned where `func` unwinds.
fn profile_on_stream<O>(
    client: &Client,
    func: impl FnOnce() -> O,
) -> Result<(O, ProfileDuration), ProfileError> {
    let mut open = OpenWindow {
        client,
        window: Some(client.profile_start()?),
    };
    let out = func();
    let window = open.take();
    client.profile_end(window).map(|duration| (out, duration))
}

/// A window opened on a client's stream, abandoned on drop unless it was taken to be closed.
struct OpenWindow<'a> {
    client: &'a Client,
    window: Option<ProfileWindow>,
}

impl OpenWindow<'_> {
    /// The window, to be closed by its taker.
    fn take(&mut self) -> ProfileWindow {
        self.window
            .take()
            .expect("an open window should be taken once")
    }
}

impl Drop for OpenWindow<'_> {
    fn drop(&mut self) {
        if let Some(window) = self.window.take() {
            self.client.profile_abandon(window);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn a_caught_panic_answers_with_its_message() {
        let caught = caught(|| panic!("the block divides the vector"));

        assert!(
            matches!(caught, Err(ServerError::Generic { reason, .. }) if reason == "the block divides the vector")
        );
    }

    #[test]
    fn a_tune_holds_its_device_unless_told_otherwise() {
        assert_eq!(TuneDeviceAccess::default(), TuneDeviceAccess::Exclusive);
    }
}

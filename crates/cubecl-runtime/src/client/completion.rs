//! A handle on the work a client's stream submitted up to one call.
//!
//! The boundary is fixed when [`Server::completion`](crate::server::Server::completion)
//! records it on the device thread, so later submissions to the same stream are
//! not part of the handle. The slot below is what makes the caller's side wait
//! without blocking that thread: the queued task fills it with the backend's
//! completion, or with the error the backend returned, and closes a channel
//! every waiter listens on.

use crate::{
    client::Client,
    server::{BackendCompletion, ServerError},
};
use alloc::{string::String, sync::Arc};
use cubecl_environment::future::{
    block_on,
    channel::{self, Receiver, Sender},
};
use cubecl_environment::sync::Mutex;

/// A handle on the work submitted to a client's stream so far.
///
/// Returned by [`Client::completion`]. The boundary is fixed at that call:
/// work submitted afterwards is not part of the handle, even if it is already
/// running when the handle is waited on. Cheap to clone; a clone names the same
/// boundary, and the device resources are released once the last clone drops.
///
/// The handle is [`Send`] and [`Sync`]: it can be waited on repeatedly and from
/// several threads at once.
#[derive(Clone)]
pub struct Completion {
    shared: Arc<CompletionShared>,
}

impl core::fmt::Debug for Completion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Completion").finish_non_exhaustive()
    }
}

impl Completion {
    /// Queue a boundary at the work `client`'s stream has submitted so far.
    ///
    /// Non-blocking: like a launch, the boundary is recorded on the device
    /// thread. Until that happens, [`is_complete`](Self::is_complete) reports
    /// `Ok(false)` — an unrecorded boundary is not complete.
    pub(crate) fn new(client: &Client) -> Self {
        let (signal, done) = channel::bounded::<()>(1);
        let shared = Arc::new(CompletionShared {
            state: Mutex::new(CompletionState {
                result: None,
                signal: Some(signal),
            }),
            done,
        });
        let recorder = CompletionRecorder {
            shared: shared.clone(),
            recorded: false,
        };
        let stream_id = client.stream_id();

        client.device.submit(move |server| {
            recorder.record(
                server
                    .completion(stream_id)
                    .map(Arc::<dyn BackendCompletion>::from),
            );
        });
        // The task has to reach the device thread even when the caller only
        // waits: a launch can sit in the client's queue until the next blocking
        // call, and a completion handle has no next call to rely on.
        client.device.flush_queue();

        Self { shared }
    }

    /// Whether the device has passed this handle's boundary.
    ///
    /// Non-blocking. `Ok(false)` means the boundary has not been recorded yet,
    /// or the device has not reached it; it does not mean the work failed.
    ///
    /// # Errors
    ///
    /// The error the backend returned when it could not record the boundary, or
    /// the error the wait on the device found.
    pub fn is_complete(&self) -> Result<bool, ServerError> {
        match self.shared.result() {
            Some(Ok(completion)) => completion.is_complete(),
            Some(Err(err)) => Err(err),
            None => Ok(false),
        }
    }

    /// Block until the device has passed this handle's boundary.
    ///
    /// Repeatable, and safe to call concurrently: every caller waits on the same
    /// boundary, and one that arrives after it was reached returns at once.
    ///
    /// # Errors
    ///
    /// The error the backend returned when it could not record the boundary, or
    /// the error the wait on the device found. A device that goes away before
    /// the boundary is recorded wakes every waiter with an error.
    pub fn wait(&self) -> Result<(), ServerError> {
        if let Some(result) = self.shared.result() {
            return finish(result);
        }
        // The channel never carries a value: closing it is the signal, and a
        // close is persistent, so one await is enough however many records race
        // with it.
        let _ = block_on(self.shared.done.recv());
        match self.shared.result() {
            Some(result) => finish(result),
            None => Err(dropped()),
        }
    }
}

type CompletionResult = Result<Arc<dyn BackendCompletion>, ServerError>;

struct CompletionShared {
    state: Mutex<CompletionState>,
    /// Closed once the queued task has filled `result`; a closed receiver is
    /// what wakes every waiter without sending a value to each of them.
    done: Receiver<()>,
}

struct CompletionState {
    result: Option<CompletionResult>,
    /// Taken and closed by [`CompletionShared::record`], which is also what
    /// wakes the waiters.
    signal: Option<Sender<()>>,
}

impl CompletionShared {
    fn result(&self) -> Option<CompletionResult> {
        self.state.lock().result.clone()
    }

    fn record(&self, result: CompletionResult) {
        let signal = {
            let mut state = self.state.lock();
            state.result = Some(result);
            state.signal.take()
        };
        if let Some(signal) = signal {
            signal.close();
        }
    }
}

/// The device-thread half of a [`Completion`]: it fills the slot, and fills it
/// with an error when it is dropped unrecorded — a device thread shut down
/// before the boundary was recorded leaves waiters nothing to wait for.
struct CompletionRecorder {
    shared: Arc<CompletionShared>,
    recorded: bool,
}

impl CompletionRecorder {
    fn record(mut self, result: CompletionResult) {
        self.recorded = true;
        self.shared.record(result);
    }
}

impl Drop for CompletionRecorder {
    fn drop(&mut self) {
        if !self.recorded {
            self.shared.record(Err(dropped()));
        }
    }
}

fn finish(result: CompletionResult) -> Result<(), ServerError> {
    match result {
        Ok(completion) => completion.wait(),
        Err(err) => Err(err),
    }
}

fn dropped() -> ServerError {
    ServerError::Generic {
        reason: String::from("the device was shut down before it recorded this completion"),
        backtrace: cubecl_environment::backtrace::BackTrace::capture(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_environment::sync::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct Done(AtomicBool);

    impl BackendCompletion for Done {
        fn is_complete(&self) -> Result<bool, ServerError> {
            Ok(self.0.load(Ordering::Acquire))
        }

        fn wait(&self) -> Result<(), ServerError> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }

    fn slot() -> (Arc<CompletionShared>, CompletionRecorder) {
        let (signal, done) = channel::bounded::<()>(1);
        let shared = Arc::new(CompletionShared {
            state: Mutex::new(CompletionState {
                result: None,
                signal: Some(signal),
            }),
            done,
        });
        let recorder = CompletionRecorder {
            shared: shared.clone(),
            recorded: false,
        };
        (shared, recorder)
    }

    /// An unrecorded boundary is not complete, and asking does not block.
    #[test]
    fn an_unrecorded_completion_is_not_complete() {
        let (shared, recorder) = slot();
        let completion = Completion { shared };

        assert!(!completion.is_complete().unwrap());
        recorder.record(Ok(Arc::new(Done(AtomicBool::new(false)))));
        assert!(!completion.is_complete().unwrap());
        completion.wait().unwrap();
        assert!(completion.is_complete().unwrap());
    }

    /// A device thread that goes away before recording wakes waiters already
    /// blocked on the slot with an error rather than leaving them waiting
    /// forever.
    #[test]
    fn a_dropped_recorder_wakes_waiters_with_an_error() {
        let (shared, recorder) = slot();
        let completion = Completion { shared };
        let waiters = 4;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(waiters + 1));

        std::thread::scope(|scope| {
            for _ in 0..waiters {
                let completion = completion.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    assert!(completion.wait().is_err());
                    assert!(completion.is_complete().is_err());
                });
            }

            barrier.wait();
            // Let the waiters reach the channel before the slot is told the
            // device went away.
            std::thread::sleep(std::time::Duration::from_millis(20));
            drop(recorder);
        });

        assert!(completion.is_complete().is_err());
    }

    /// The error a backend returned reaches both methods of the handle.
    #[test]
    fn a_recorded_error_reaches_the_handle() {
        let (shared, recorder) = slot();
        let completion = Completion { shared };
        recorder.record(Err(ServerError::unsupported("completion")));

        assert!(completion.is_complete().is_err());
        assert!(completion.wait().is_err());
    }
}

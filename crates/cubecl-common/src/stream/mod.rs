//! Pluggable executors that decide *where* a stream's work runs.
//!
//! A [`StreamExecutor`] runs a closure to completion, blocking the caller until
//! it returns. The closure itself is responsible for scoping the stream id (see
//! [`StreamId::executes`](crate::stream_id::StreamId::executes)); the executor
//! only decides on which thread the closure runs.
//!
//! * [`InlineExecutor`] runs the closure on the calling thread. It is always
//!   available, including on wasm, and lets many threads share one stream/pool.
//! * [`ThreadExecutor`] owns a single worker thread and serializes every closure
//!   onto it (mirroring the `ChannelDeviceHandle` worker pattern). Available only
//!   when real threads exist (`std`, non-wasm).

use alloc::boxed::Box;
use core::any::Any;

/// A type-erased job: runs to completion and returns its result boxed.
///
/// Erasing the closure (rather than making the trait method generic) keeps
/// [`StreamExecutor`] object-safe so it can live behind `Arc<dyn StreamExecutor>`.
pub type ErasedJob<'a> = Box<dyn FnOnce() -> Box<dyn Any + Send> + Send + 'a>;

/// Decides where a stream's work runs.
///
/// Modeled on the `DeviceHandleSpec` "run a borrowing `FnOnce` and block on the
/// result" surface. Implementations must run the job to completion before
/// returning and propagate any panic the job raised.
pub trait StreamExecutor: Send + Sync + 'static {
    /// Runs `job` to completion, blocking the caller, and returns its result.
    ///
    /// If `job` panics, the panic is propagated to the caller.
    fn run_blocking(&self, job: ErasedJob<'_>) -> Box<dyn Any + Send>;
}

/// Executor that runs the job on the calling thread.
///
/// Always available (including wasm). When several threads hold a clone of the
/// same `Stream`, they all run inline and share the stream's id — hence one
/// fusion queue and one cubecl memory pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlineExecutor;

impl StreamExecutor for InlineExecutor {
    fn run_blocking(&self, job: ErasedJob<'_>) -> Box<dyn Any + Send> {
        job()
    }
}

#[cfg(multi_threading)]
pub use thread::ThreadExecutor;

#[cfg(multi_threading)]
mod thread {
    use super::*;
    use std::{
        boxed::Box,
        panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
        sync::mpsc,
        thread::{self, JoinHandle},
    };

    /// A `'static` closure enqueued on the worker thread.
    type Work = Box<dyn FnOnce() + Send + 'static>;

    /// Executor backed by a single dedicated worker thread.
    ///
    /// Every job is serialized onto the worker and the caller blocks until it
    /// returns, reusing the same borrowing-`FnOnce` shim as the device channel:
    /// a job that borrows from the caller's stack is sent to the worker through
    /// a raw pointer and kept alive by blocking on the result channel.
    pub struct ThreadExecutor {
        sender: Option<mpsc::Sender<Work>>,
        worker: Option<JoinHandle<()>>,
    }

    impl ThreadExecutor {
        /// Spawns the worker thread.
        pub fn new() -> Self {
            let (sender, receiver) = mpsc::channel::<Work>();
            let worker = thread::Builder::new()
                .name("cubecl-stream".into())
                .spawn(move || {
                    // Exits when the sender is dropped (see `Drop`).
                    while let Ok(work) = receiver.recv() {
                        work();
                    }
                })
                .expect("Unable to spawn the stream executor worker thread");

            Self {
                sender: Some(sender),
                worker: Some(worker),
            }
        }
    }

    impl Default for ThreadExecutor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ThreadExecutor {
        fn drop(&mut self) {
            // Close the channel so the worker's `recv` loop ends, then join it.
            // `run_blocking` is synchronous, so the worker is idle by this point.
            self.sender = None;
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    impl StreamExecutor for ThreadExecutor {
        fn run_blocking(&self, job: ErasedJob<'_>) -> Box<dyn Any + Send> {
            let (result_tx, result_rx) = mpsc::channel();

            // Run `job` under `catch_unwind` on the worker so a panic is captured
            // and re-raised on the caller instead of aborting the worker.
            let mut slot = Some(move || {
                let _ = result_tx.send(catch_unwind(AssertUnwindSafe(job)));
            });

            let sender = self
                .sender
                .as_ref()
                .expect("sender is only cleared on drop");
            sender
                .send(create_shim(&mut slot))
                .expect("the stream executor worker thread is alive");

            match result_rx.recv() {
                Ok(Ok(value)) => value,
                Ok(Err(payload)) => resume_unwind(payload),
                Err(_) => panic!("the stream executor worker thread disconnected"),
            }
        }
    }

    /// Builds a `'static` shim that consumes `*slot` on the worker thread.
    ///
    /// The caller must keep `*slot` alive until the shim has run; `run_blocking`
    /// does this by blocking on the result channel. This mirrors the
    /// `ChannelDeviceHandle::run_scoped` shim so a borrowing `FnOnce` can run on
    /// another thread.
    fn create_shim<W: FnOnce() + Send>(slot: &mut Option<W>) -> Work {
        // `*mut ()` so the shim is `'static`.
        struct Ptr(*mut ());
        // SAFETY: the pointee is `Send` by the bound on `W`; unique access is
        // upheld by the single deref below.
        unsafe impl Send for Ptr {}

        let ptr = Ptr(slot as *mut _ as *mut ());
        Box::new(move || {
            let _ = &ptr; // capture the whole `Ptr` so the closure is `Send`.
            // SAFETY:
            // - The caller keeps `*slot` alive through the shim's run.
            // - The shim is `FnOnce`, run at most once, so `*slot` is `Some`.
            // `Option::take` flips `*slot` to `None`, keeping drop correct.
            let work = unsafe { (*(ptr.0 as *mut Option<W>)).take().unwrap_unchecked() };
            work()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper that runs a non-erased closure through an executor and downcasts.
    fn run<R: Send + 'static>(exec: &dyn StreamExecutor, f: impl FnOnce() -> R + Send) -> R {
        let job: ErasedJob<'_> = Box::new(move || Box::new(f()) as Box<dyn Any + Send>);
        *exec.run_blocking(job).downcast::<R>().unwrap()
    }

    #[test]
    fn inline_executor_runs_job_and_returns_result() {
        let value = 21;
        let out = run(&InlineExecutor, move || value * 2);
        assert_eq!(out, 42);
    }

    #[cfg(multi_threading)]
    #[test]
    fn thread_executor_runs_job_and_returns_result() {
        let exec = ThreadExecutor::new();
        // `value` is borrowed from this stack: proves the borrowing-FnOnce shim works.
        let value = 20;
        let out = run(&exec, || value + 1);
        assert_eq!(out, 21);
    }

    #[cfg(multi_threading)]
    #[test]
    fn thread_executor_runs_on_a_different_thread() {
        let exec = ThreadExecutor::new();
        let caller = std::thread::current().id();
        let worker = run(&exec, move || std::thread::current().id());
        assert_ne!(caller, worker, "the job must run on the worker thread");
    }

    #[cfg(multi_threading)]
    #[test]
    fn thread_executor_propagates_panic() {
        let exec = ThreadExecutor::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run::<()>(&exec, || panic!("boom"));
        }));
        let payload = result.expect_err("the worker panic must propagate to the caller");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("boom"),
            "the original panic message must be preserved"
        );

        // The worker survives a propagated panic and still runs later jobs.
        let out = run(&exec, || 7);
        assert_eq!(out, 7);
    }
}

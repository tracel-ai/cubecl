//! Channel types shared across environments.
//!
//! Re-exports the channels used throughout `CubeCL` so consumers don't depend on
//! the underlying crates directly.

pub use async_channel::{
    Receiver, RecvError, SendError, Sender, TryRecvError, TrySendError, bounded, unbounded,
};

/// One-shot channels: a single value handed from one task to another.
///
/// # Blocking
///
/// [`Receiver::recv`] parks the calling thread, which on wasm is the browser's
/// only thread: it hangs the event loop with no diagnostic, unlike
/// [`block_on`](super::block_on) and [`read_sync`](super::read_sync), which
/// handle that target explicitly. Await the receiver instead — that works
/// everywhere. The blocking receive is only used from `multi_threading` code
/// paths.
///
/// Absent on a target whose atomics carry no compare-and-swap (thumbv6m):
/// `oneshot` needs it and has no fallback. Every consumer needs `std` or the
/// browser, so nothing that could run there loses anything.
#[cfg(target_has_atomic = "ptr")]
pub mod oneshot {
    // A glob because the error types are conditionally compiled on the
    // underlying crate's features; naming them would break feature combinations
    // rather than document them.
    pub use ::oneshot::*;
}

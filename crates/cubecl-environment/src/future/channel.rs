//! Channel types shared across environments.
//!
//! Re-exports the channels used throughout `CubeCL` so consumers don't depend on
//! the underlying crates directly.

pub use async_channel::{
    Receiver, RecvError, SendError, Sender, TryRecvError, TrySendError, bounded, unbounded,
};

pub use oneshot;

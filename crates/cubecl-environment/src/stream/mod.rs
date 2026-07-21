//! Stream identity, policies and manual stream management.
//!
//! A stream is a logical submission queue: work on one stream executes in
//! order, work on different streams may overlap. This module owns how the
//! *current* stream is derived:
//!
//! - By default, every OS thread gets its own stream ([`StreamPolicy::PerThread`]).
//! - Under tokio, [`StreamPolicy::PerTask`] keeps a task's stream stable across
//!   `.await` points even when the executor moves it between worker threads.
//! - [`Stream`] gives manual control: `Stream::spawn`, `Stream::enter` and
//!   `Stream::attach` pin work to an explicit stream in any environment.

mod handle;
mod id;
mod policy;

pub use handle::*;
pub use id::StreamId;
pub use policy::*;

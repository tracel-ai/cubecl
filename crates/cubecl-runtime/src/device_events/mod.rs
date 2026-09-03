//! Device events, and the two things built on them.
//!
//! The C-family device APIs — CUDA and HIP — expose the same event model, and
//! the backends over them were each writing the same two types against it: a
//! fence handed out so a caller can wait on a stream outside the server's lock,
//! and a profiler that times work on the device's own clock. [`EventApi`] is
//! the seam: a backend spells the driver calls, and [`EventFence`] and
//! [`EventProfiler`] are written once above them.

mod base;
mod event;
mod fence;
mod profiler;

pub use base::*;
pub use event::*;
pub use fence::*;
pub use profiler::*;

//! Device events, and what is built on them.
//!
//! The C-family device APIs — CUDA and HIP — expose the same event model, and
//! the backends over them were each writing the same code against it.
//! [`EventApi`] is the seam: a backend spells the driver calls, and what sits
//! above them is written once.

mod base;
mod event;
mod profiler;

pub use base::*;
pub use event::*;
pub use profiler::*;

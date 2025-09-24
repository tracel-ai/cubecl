/// Scheduler based multi-stream support.
pub mod scheduler;

mod base;

#[cfg(multi_threading)]
mod event;

pub use base::*;

#[cfg(multi_threading)]
pub use event::*;

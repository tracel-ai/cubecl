/// Scheduler based multi-stream support.
pub mod scheduler;

mod base;
mod handle;

#[cfg(multi_threading)]
mod event;

pub use base::*;
pub use handle::Stream;

// Re-export the executor surface so a `Stream` can be built and customized
// without reaching into `cubecl_common`.
#[cfg(multi_threading)]
pub use cubecl_common::stream::ThreadExecutor;
pub use cubecl_common::stream::{InlineExecutor, StreamExecutor};

#[cfg(multi_threading)]
pub use event::*;

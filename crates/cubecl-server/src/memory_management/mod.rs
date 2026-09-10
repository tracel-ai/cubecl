//! Memory pools and the bookkeeping behind a [`Handle`](crate::server::Handle).
//!
//! The value types a client sees — reports, configuration, the handle itself —
//! are `cubecl-runtime`'s and re-exported here.

pub use cubecl_runtime::memory_management::*;

pub(crate) mod memory_pool;

mod error_graph;
mod taint;

/// Export utilities to keep track of CPU buffers when performing async data copies.
#[cfg(multi_threading)]
pub mod drop_queue;

pub use error_graph::*;
pub use taint::*;

/// Dynamic memory management strategy.
mod memory_manage;
pub use memory_manage::*;

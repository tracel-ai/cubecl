//! Memory pools and the bookkeeping behind a [`Handle`](crate::server::Handle).
//!
//! The value types a client sees — reports, configuration, the handle itself —
//! are `cubecl-runtime`'s and re-exported here.

pub use cubecl_runtime::memory_management::*;

pub(crate) mod memory_pool;

mod error_graph;
mod failure_id;
mod taint;

/// Holding host memory, and memory of other devices, until the device is done reading it.
#[cfg(multi_threading)]
pub mod drop_queue;

pub use error_graph::*;
pub use failure_id::*;
pub use taint::*;

mod dynamic;
pub(crate) use dynamic::*;

/// Dynamic memory management strategy.
mod memory_manage;
pub use memory_manage::*;

mod options;
pub use options::*;

/// Moving live allocations off outdated pages.
pub mod relocation;

mod release;
pub(crate) use release::release_or_leak;

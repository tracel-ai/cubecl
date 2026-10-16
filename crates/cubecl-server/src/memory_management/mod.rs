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

mod dynamic;
pub(crate) use dynamic::*;

/// Dynamic memory management strategy.
mod memory_manage;
pub use memory_manage::*;

/// Moving live allocations off outdated pages.
pub mod relocation;

/// Release memory after a failed wait only when the device is poisoned and can no longer use it.
/// Otherwise completion is unknown, so keep the allocations reserved by leaking their owners.
pub(crate) fn release_or_leak<T>(memory: T, err: &crate::server::ServerError) {
    if err.is_device_poisoned() {
        core::mem::drop(memory);
    } else {
        core::mem::forget(memory);
        log::error!("leaking memory the device may still be using, after a failed wait: {err}");
    }
}

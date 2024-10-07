pub(crate) mod memory_pool;

mod base;
mod memory_lock;

pub use base::*;
pub use memory_lock::*;

/// Dynamic memory management strategy.
pub mod dynamic;
/// Simple memory management strategy.
pub mod simple;

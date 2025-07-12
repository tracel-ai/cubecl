//! Loaders read from global memory and write to stage memory
//!
//! Loaders fall into two axes:
//!
//! - **Synchronization**
//!   - **Synchronous**: Performs direct memory accesses.
//!   - **Asynchronous**: Uses `memcpy_async` for loading.
//!
//! - **Coverage**
//!   - **Full**: Loads the entire shared memory region.
//!   - **Partial**: Loads only a single stage, when multiple stages share the same memory.

mod loader;
mod strategy;

pub use loader::*;
pub use strategy::*;

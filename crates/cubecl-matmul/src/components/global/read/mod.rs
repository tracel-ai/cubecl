//! Readers read from global memory and write to stage memory
//!
//! Readers fall into two axes:
//!
//! - **Synchronization**
//!   - **Synchronous**: Performs direct memory accesses.
//!   - **Asynchronous**: Uses `memcpy_async` for reading.
//!
//! - **Coverage**
//!   - **Full**: Reads the entire shared memory region.
//!   - **Partial**: Reads only a single stage, when multiple stages share the same memory.

mod layout;
mod reader;
mod strategy;

pub use layout::*;
pub use reader::*;
pub use strategy::*;

//! Executes multiple independent global matmuls with optional broadcasting.

mod base;
mod entry_point;
mod partitioned_matmul;

pub use base::*;
pub use partitioned_matmul::*;

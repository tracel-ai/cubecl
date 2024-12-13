/// Matmul using Accelerator or PlaneMma
pub mod matmul;
/// Simple non-cooperative matmul that can be very fast on small matrices.
pub mod simple;
/// Non-cooperative Matmul
pub mod tiling2d;

mod error;

pub use error::*;

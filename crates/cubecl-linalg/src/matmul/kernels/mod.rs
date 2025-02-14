/// Matmul using Accelerator or PlaneMma
pub mod matmul;
/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;
/// Non-cooperative Matmul
pub mod tiling2d;

mod error;

pub use error::*;

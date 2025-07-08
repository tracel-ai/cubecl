/// Matmul using Accelerator or PlaneMma
pub mod layered;
/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;

mod error;

pub use error::*;

//! This provides different implementations of the associative scan algorithm
//! which can run on multiple GPU backends using CubeCL.
//!
//! The commonly known prefix sum or cumsum operation is an associative scan
//! using the associative addition operator. In general, an associative scan
//! is a (parallel) scan operation with an operator that is required to be
//! associative with the following property:
//! * Let the input sequence of numbers be `x_0`, `x_1`, `x_2`, ...
//! * Let the output sequence of numbers by `y_0`, `y_1`, `y_2`, ...
//! * The output is now defined as `y_0 = x_0`, `y_1 = x_0 + x_1`,
//!   `y_2 = x_0 + x_1 + x_2`, ...

mod base;
mod config;
mod error;
pub mod instructions;
pub mod kernels;

#[cfg(feature = "export_tests")]
pub mod tests;

pub use base::*;
pub use config::*;
pub use error::*;

use crate::instructions::ScanInstruction;
use cubecl_core::prelude::*;

// ToDo: add algorithm reference to the book
// ToDo: write benchmarks (the algorithm should be almost 100% memory bound ideally)
// ToDo: abstract the algorithm selection into a strategy enum like in matmul

pub fn associative_scan<R: Runtime, N: Numeric, I: ScanInstruction>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<'_, R>,
    output: TensorHandleRef<'_, R>,
    axis: usize,
    inclusive: bool,
) -> Result<(), ScanError> {
    // ToDo: maybe allocate the secondary storage using client here
    // ToDo: at least in CUDA, the kernels launched here should execute sequentially => important for the 3-stage impl

    kernels::decoupled_lookback::launch_ref::<R, N, I>(client, &input, &output, axis, inclusive)?;

    Ok(())
}

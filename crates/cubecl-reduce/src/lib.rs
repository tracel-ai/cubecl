mod config;
mod error;
mod instructions;
mod launch;
mod strategy;

pub use config::*;
pub use error::*;
pub use instructions::*;
pub use launch::*;
pub use strategy::*;

#[cfg(feature = "export_tests")]
pub mod test;

use cubecl_core::prelude::*;

/// Entry point for reduce.
pub fn reduce<R: Runtime, In: Numeric, Out: Numeric, Inst: Reduce<In>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    axis: usize,
    strategy: Option<ReduceStrategy>,
) -> Result<(), ReduceError> {
    validate_axis(input.shape.len(), axis)?;
    valide_output_shape(input.shape, output.shape, axis)?;
    let strategy = strategy
        .map(|s| s.validate::<R>(client))
        .unwrap_or(Ok(ReduceStrategy::fallback_strategy::<R>(client)))?;
    let config = ReduceConfig::generate(client, &input, &output, axis, &strategy);
    launch_reduce::<R, In, Out, Inst>(client, input, output, axis as u32, config, strategy);
    Ok(())
}

// Check that the given axis is less than the rank of the input.
fn validate_axis(rank: usize, axis: usize) -> Result<(), ReduceError> {
    if axis > rank {
        return Err(ReduceError::InvalidAxis { axis, rank });
    }
    Ok(())
}

// Check that the output shape match the input shape with the given axis set to 1.
fn valide_output_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> Result<(), ReduceError> {
    let mut expected_shape = input_shape.to_vec();
    expected_shape[axis] = 1;
    if output_shape != expected_shape {
        return Err(ReduceError::MismatchShape {
            expected_shape,
            output_shape: output_shape.to_vec(),
        });
    }
    Ok(())
}

//! This provides different implementations of the reduce algorithm which
//! can run on multiple GPU backends using CubeCL.
//!
//! A reduction is a tensor operation mapping a rank `R` tensor to a rank `R - 1`
//! by agglomerating all elements along a given axis with some binary operator.
//! This is often also called folding.
//!
//! This crate provides a main entrypoint as the [`reduce`] function which allows to automatically
//! perform a reduction for a given instruction implementing the [`ReduceInstruction`] trait and a given [`ReduceStrategy`].
//! It also provides implementation of the [`ReduceInstruction`] trait for common operations in the [`instructions`] module.
//! Finally, it provides many reusable primitives to perform different general reduction algorithms in the [`primitives`] module.

pub mod instructions;
pub mod primitives;

mod config;
mod error;
mod launch;
mod strategy;

pub use config::*;
pub use error::*;
pub use instructions::Reduce;
pub use instructions::ReduceInstruction;
pub use strategy::*;

use launch::*;

#[cfg(feature = "export_tests")]
pub mod test;

use cubecl_core::prelude::*;

/// Reduce the given `axis` of the `input` tensor using the instruction `Inst` and write the result into `output`.
///
/// An optional [`ReduceStrategy`] can be provided to force the reduction to use a specific algorithm. If omitted, a best effort
/// is done to try and pick the best strategy supported for the provided `client`.
///
/// Return an error if `strategy` is `Some(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
/// The shape of `output` must be the same as input except with a value of 1 for the given `axis`.
///
///
/// # Example
///
/// This examples show how to sum the rows of a small `2 x 2` matrix into a `1 x 2` vector.
/// For more details, see the CubeCL documentation.
///
/// ```ignore
/// use cubecl_reduce::instructions::Sum;
///
/// let client = /* ... */;
/// let size_f32 = std::mem::size_of::<f32>();
/// let axis = 0; // 0 for rows, 1 for columns in the case of a matrix.
///
/// // Create input and output handles.
/// let input_handle = client.create(f32::as_bytes(&[0, 1, 2, 3]));
/// let input = unsafe {
///     TensorHandleRef::<R>::from_raw_parts(
///         &input_handle,
///         &[2, 1],
///         &[2, 2],
///         size_f32,
///     )
/// };
///
/// let output_handle = client.empty(2 * size_f32);
/// let output = unsafe {
///     TensorHandleRef::<R>::from_raw_parts(
///         &output_handle,
///         &output_stride,
///         &output_shape,
///         size_f32,
///     )
/// };
///
/// // Here `R` is a `cubecl::Runtime`.
/// let result = reduce::<R, f32, f32, Sum>(&client, input, output, axis, None);
///
/// if result.is_ok() {
///        let binding = output_handle.binding();
///        let bytes = client.read_one(binding);
///        let output_values = f32::from_bytes(&bytes);
///        println!("Output = {:?}", output_values); // Should print [1, 5].
/// }
/// ```
pub fn reduce<R: Runtime, In: Numeric, Out: Numeric, Inst: Reduce>(
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
    let config = ReduceConfig::generate::<R, In>(client, &input, &output, axis, &strategy);

    if let CubeCount::Static(x, y, z) = config.cube_count {
        let (max_x, max_y, max_z) = R::max_cube_count();
        if x > max_x || y > max_y || z > max_z {
            return Err(ReduceError::CubeCountTooLarge);
        }
    }

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

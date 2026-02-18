#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[macro_use]
extern crate derive_new;

pub use cubecl_zspace as zspace;

/// Cube Frontend Types.
pub mod frontend;
/// Input Output utilities.
pub mod io;

pub mod post_processing;

/// Some future utilities that work across environments.
pub use cubecl_common::future;

use cubecl_ir::LineSize;
use cubecl_runtime::client::ComputeClient;
pub use cubecl_runtime::memory_management::MemoryConfiguration;
use cubecl_runtime::server::CubeCountSelection;
pub use frontend::cmma;

/// Cube Language Internal Representation.
pub use cubecl_ir as ir;

pub mod codegen;
pub mod compute;
pub mod prelude;

mod pod;

pub use codegen::*;
pub use cubecl_runtime::runtime::*;
pub use pod::*;

pub use cubecl_macros::*;
pub use cubecl_runtime::benchmark;
pub use cubecl_runtime::client;
pub use cubecl_runtime::compiler::{CompilationError, Compiler, CubeTask};
pub use cubecl_runtime::memory_management::MemoryUsage;
pub use cubecl_runtime::server;
pub use cubecl_runtime::tune;

use frontend::LaunchArg;

pub use cubecl_common::*;

pub use prelude::CubeCount;
pub use prelude::{CubeDim, ExecutionMode};

pub use num_traits;

mod id;
pub use id::*;

// Private utils for macros
#[doc(hidden)]
pub mod __private {
    pub use alloc::{format, vec};
}

/// Calculate the number of cubes required to execute an operation where one cube unit is
/// assigned to one element.
pub fn calculate_cube_count_elemwise<R: Runtime>(
    client: &ComputeClient<R>,
    num_elems: usize,
    cube_dim: CubeDim,
) -> CubeCount {
    let num_cubes = num_elems.div_ceil(cube_dim.num_elems() as usize);
    CubeCountSelection::new(client, num_cubes as u32).cube_count()
}

pub fn tensor_vectorization_factor(
    factors: &[LineSize],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
) -> LineSize {
    tensor_line_size_parallel(factors.iter().cloned(), shape, strides, dim)
}
pub fn tensor_line_size(
    factors: &[LineSize],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
) -> LineSize {
    tensor_line_size_parallel(factors.iter().cloned(), shape, strides, dim)
}

#[derive(Debug, Clone)]
pub enum LineSizeError {
    AxisOutOfBounds,
    StrideMismatch,
    NoValidLineSize,
}

/// Find the maximum line size usable for parallel vectorization along the given axis
/// from the supported line sizes or return 1 if vectorization is impossible.
///
/// This function is designed to never return a line size above 1 by error,
/// but doesn't guarantee to always return the actual maximum possible line size.
/// That is, it may be overly strict.
///
/// Currently, this checks that the stride of the axis is 1, that it's shape is
/// divisible by a candidate line size and that the smallest stride that is not 1
/// is divisible by the vectorization.
/// The last condition ensure that the current axis is contiguous within the next stride.
pub fn tensor_line_size_parallel(
    optimized_line_sizes: impl Iterator<Item = LineSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> LineSize {
    try_tensor_line_size_parallel(optimized_line_sizes, shape, strides, axis).unwrap_or(1)
}

/// Like `try_tensor_line_size_parallel` but does not assume 1 is supported
pub fn try_tensor_line_size_parallel(
    supported_line_sizes: impl Iterator<Item = LineSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> Result<LineSize, LineSizeError> {
    let stride = strides.get(axis).ok_or(LineSizeError::AxisOutOfBounds)?;
    if *stride != 1 {
        return Err(LineSizeError::StrideMismatch);
    }

    let axis_shape = shape.get(axis).ok_or(LineSizeError::AxisOutOfBounds)?;

    let next_stride = *strides
        .iter()
        .filter(|&&stride| stride > 1)
        .min()
        .unwrap_or(&0);

    supported_line_sizes
        .filter(|&line_size| axis_shape % line_size == 0 && next_stride % line_size == 0)
        .max()
        .ok_or(LineSizeError::NoValidLineSize)
}

/// Find the maximum line size usable for perpendicular vectorization along the given axis
/// from the supported line sizes or return 1 if vectorization is impossible.
///
/// This function is designed to never return a line size above 1 by error,
/// but doesn't guarantee to always return the actual maximum possible line size.
/// That is, it may be overly strict.
///
/// Currently, this checks that the stride of the axis is divisible by a candidate line size
/// and that the product of all shapes of axes with smaller strides is equal to the stride of the axis.
/// The second condition ensure that elements within the stride are contiguous.
pub fn tensor_line_size_perpendicular(
    supported_line_sizes: impl Iterator<Item = LineSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> LineSize {
    try_tensor_line_size_perpendicular(supported_line_sizes, shape, strides, axis).unwrap_or(1)
}

/// Like `tensor_line_size_perpendicular` but does not assume 1 is supported
pub fn try_tensor_line_size_perpendicular(
    supported_line_sizes: impl Iterator<Item = LineSize>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> Result<LineSize, LineSizeError> {
    let axis_stride = strides.get(axis).ok_or(LineSizeError::AxisOutOfBounds)?;

    let prod_shape_axes_smaller_strides = strides
        .iter()
        .zip(shape.iter())
        .filter(|(stride, _)| **stride < *axis_stride)
        .map(|(_, shape)| shape)
        .product::<usize>();

    if *axis_stride != prod_shape_axes_smaller_strides {
        return Err(LineSizeError::StrideMismatch);
    }

    supported_line_sizes
        .filter(|&line_size| *axis_stride % line_size == 0)
        .max()
        .ok_or(LineSizeError::NoValidLineSize)
}

/// Runtime arguments to launch a kernel.
pub type RuntimeArg<'a, T, R> = <T as LaunchArg>::RuntimeArg<'a, R>;

#[cfg(feature = "export_tests")]
/// Tests only useful for runtimes.
pub mod runtime_tests;

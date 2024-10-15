extern crate alloc;

#[macro_use]
extern crate derive_new;

/// Cube Frontend Types.
pub mod frontend;

/// Some future utilities that work across environments.
pub use cubecl_common::future;

pub use cubecl_runtime::memory_management::MemoryConfiguration;
pub use frontend::cmma;

/// Cube Language Internal Representation.
pub mod ir;

pub mod codegen;
pub mod compute;
pub mod prelude;

mod pod;
mod runtime;

pub use codegen::*;
pub use pod::*;
pub use runtime::*;

pub use cubecl_macros::*;
pub use cubecl_runtime::benchmark;

/// An approximation of the subcube dimension.
pub const SUBCUBE_DIM_APPROX: usize = 16;

use crate::ir::KernelDefinition;
use frontend::LaunchArg;

pub use prelude::CubeCount;
pub use prelude::CubeDim;

mod id;
pub use id::*;

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait Kernel: Send + Sync + 'static + Sized {
    /// Convert to a kernel definition.
    fn define(&self) -> KernelDefinition;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Calculate the number of cubes required to execute an operation where one cube unit is
/// assigned to one element.
pub fn calculate_cube_count_elemwise(num_elems: usize, cube_dim: CubeDim) -> CubeCount {
    let num_elems_per_cube = cube_dim.num_elems();
    let cube_counts = f32::ceil(num_elems as f32 / num_elems_per_cube as f32);
    let cube_count_x = f32::ceil(f32::sqrt(cube_counts));
    let cube_count_y = f32::ceil(num_elems as f32 / (cube_count_x * num_elems_per_cube as f32));

    CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1)
}

pub fn tensor_vectorization_factor(
    factors: &[u8],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
) -> u8 {
    tensor_line_size(factors, shape, strides, dim)
}

pub fn tensor_line_size(factors: &[u8], shape: &[usize], strides: &[usize], dim: usize) -> u8 {
    match strides.get(dim) {
        Some(val) => {
            if *val != 1 {
                return 1;
            }
        }
        None => return 1,
    }

    let shape_check = match shape.get(dim) {
        Some(val) => val,
        None => return 1,
    };

    let stride_check = if let Some(dim) = dim.checked_sub(1) {
        strides.get(dim)
    } else {
        None
    };

    for factor in factors {
        let factor = *factor as usize;

        if shape_check % factor == 0 {
            match stride_check {
                Some(check) => {
                    if check % factor == 0 {
                        return factor as u8;
                    }
                }
                None => return factor as u8,
            }
        }
    }

    1
}

/// Runtime arguments to launch a kernel.
pub type RuntimeArg<'a, T, R> = <T as LaunchArg>::RuntimeArg<'a, R>;

#[cfg(feature = "export_tests")]
/// Tests only useful for runtimes.
pub mod runtime_tests;

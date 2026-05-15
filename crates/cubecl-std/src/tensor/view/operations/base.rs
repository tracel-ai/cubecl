#![allow(clippy::mut_from_ref)]

use crate::tensor::layout::Coordinates;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier, unexpanded};

/// Type from which we can read values in cube functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[cube(expand_base_traits = "VectorizedExpand")]
pub trait ViewOperations<T: CubePrimitive, C: Coordinates>: Vectorized {
    #[allow(unused)]
    fn read(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_checked(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_masked(&self, pos: C, value: T) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, pos: C) -> T {
        unexpanded!()
    }

    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused)]
    fn as_linear_slice(&self, pos: C, size: C) -> &[T] {
        unexpanded!()
    }

    /// Execute a TMA load into shared memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    #[allow(unused)]
    fn tensor_map_load(&self, barrier: &Barrier, shared_memory: &mut [T], pos: C) {
        unexpanded!()
    }

    #[allow(unused)]
    fn shape(&self) -> C {
        unexpanded!();
    }

    #[allow(unused)]
    fn is_in_bounds(&self, pos: C) -> bool {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[cube(expand_base_traits = "ViewOperationsExpand<T, C>")]
pub trait ViewOperationsMut<T: CubePrimitive, C: Coordinates>: ViewOperations<T, C> {
    #[allow(unused)]
    fn write(&self, pos: C, value: T) {
        unexpanded!()
    }

    #[allow(unused)]
    fn write_checked(&self, pos: C, value: T) {
        unexpanded!()
    }

    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused, clippy::wrong_self_convention)]
    fn as_linear_slice_mut(&self, pos: C, size: C) -> &mut [T] {
        unexpanded!()
    }

    /// Execute a TMA store into global memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    #[allow(unused)]
    fn tensor_map_store(&self, shared_memory: &[T], pos: C) {
        unexpanded!()
    }
}

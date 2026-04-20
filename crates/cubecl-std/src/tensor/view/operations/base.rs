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
    #[allow(unused, clippy::needless_lifetimes)]
    fn to_linear_slice<'a>(&'a self, pos: C, size: C) -> &'a Slice<T, ReadOnly> {
        unexpanded!()
    }

    ///.Execute a TMA load into shared memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    #[allow(unused)]
    fn tensor_map_load(&self, barrier: &Barrier, shared_memory: &mut Slice<T, ReadWrite>, pos: C) {
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
/// For an immutable version, see [List]."
pub trait ViewOperationsMut<T: CubePrimitive, C: Coordinates>:
    ViewOperations<T, C> + cubecl::prelude::CubeType<ExpandType: ViewOperationsMutExpand<T, C>>
{
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
    #[allow(
        unused,
        clippy::wrong_self_convention,
        clippy::mut_from_ref,
        clippy::needless_lifetimes
    )]
    fn to_linear_slice_mut<'a>(&'a self, pos: C, size: C) -> &'a mut Slice<T, ReadWrite> {
        unexpanded!()
    }

    /// Execute a TMA store into global memory, if the underlying storage supports it.
    /// Panics if it\'s unsupported.
    #[allow(unused)]
    fn tensor_map_store(&self, shared_memory: &Slice<T>, pos: C) {
        unexpanded!()
    }

    fn __expand_write(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
        pos: <C as CubeType>::ExpandType,
        value: <T as CubeType>::ExpandType,
    ) {
        this.__expand_write_method(scope, pos, value)
    }

    fn __expand_write_checked(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
        pos: <C as CubeType>::ExpandType,
        value: <T as CubeType>::ExpandType,
    ) {
        this.__expand_write_checked_method(scope, pos, value)
    }

    fn __expand_to_linear_slice_mut<'a>(
        scope: &Scope,
        this: &'a <Self as CubeType>::ExpandType,
        pos: <C as CubeType>::ExpandType,
        size: <C as CubeType>::ExpandType,
    ) -> &'a mut <Slice<T, ReadWrite> as CubeType>::ExpandType {
        this.__expand_to_linear_slice_mut_method(scope, pos, size)
    }
    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_store(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
        shared_memory: &<Slice<T> as CubeType>::ExpandType,
        pos: <C as CubeType>::ExpandType,
    ) {
        this.__expand_tensor_map_store_method(scope, shared_memory, pos)
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[allow(clippy::too_many_arguments)]
pub trait ViewOperationsMutExpand<T: CubePrimitive, C: Coordinates>:
    ViewOperationsExpand<T, C>
{
    fn __expand_write_method(
        &self,
        scope: &cubecl::prelude::Scope,
        pos: <C as cubecl::prelude::CubeType>::ExpandType,
        value: <T as cubecl::prelude::CubeType>::ExpandType,
    ) -> ();

    fn __expand_write_checked_method(
        &self,
        scope: &cubecl::prelude::Scope,
        pos: <C as cubecl::prelude::CubeType>::ExpandType,
        value: <T as cubecl::prelude::CubeType>::ExpandType,
    ) -> ();

    #[allow(clippy::mut_from_ref)]
    fn __expand_to_linear_slice_mut_method<'a>(
        &'a self,
        scope: &Scope,
        pos: <C as CubeType>::ExpandType,
        size: <C as CubeType>::ExpandType,
    ) -> &'a mut <Slice<T, ReadWrite> as CubeType>::ExpandType;

    fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &<Slice<T> as CubeType>::ExpandType,
        pos: <C as CubeType>::ExpandType,
    ) -> ();
}

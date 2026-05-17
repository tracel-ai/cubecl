#![allow(clippy::mut_from_ref)]

use crate::tensor::layout::Coordinates;
use cubecl_core::{
    self as cubecl,
    frontend::{NativeExpand, barrier::Barrier},
    prelude::*,
    unexpanded,
};

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
    #[allow(unused)]
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

macro_rules! view_ops_read_ref {
    ($ty: ty) => {
        impl<T: CubePrimitive, C: Coordinates, V: ViewOperations<T, C> + ?Sized>
            ViewOperations<T, C> for $ty
        {
        }
        impl<T: CubePrimitive, C: Coordinates, V: ViewOperationsExpand<T, C> + ?Sized>
            ViewOperationsExpand<T, C> for $ty
        {
            #[allow(clippy::too_many_arguments)]
            fn __expand_read_method(&self, scope: &Scope, pos: C::ExpandType) -> T::ExpandType {
                (**self).__expand_read_method(scope, pos)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_read_checked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> T::ExpandType {
                (**self).__expand_read_checked_method(scope, pos)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_read_masked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
                value: T::ExpandType,
            ) -> T::ExpandType {
                (**self).__expand_read_masked_method(scope, pos, value)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_read_unchecked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> T::ExpandType {
                (**self).__expand_read_unchecked_method(scope, pos)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_as_linear_slice_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
                size: C::ExpandType,
            ) -> &SliceExpand<T> {
                (**self).__expand_as_linear_slice_method(scope, pos, size)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_tensor_map_load_method(
                &self,
                scope: &Scope,
                barrier: &NativeExpand<Barrier>,
                shared_memory: &mut SliceExpand<T>,
                pos: C::ExpandType,
            ) -> () {
                (**self).__expand_tensor_map_load_method(scope, barrier, shared_memory, pos);
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_shape_method(&self, scope: &Scope) -> C::ExpandType {
                (**self).__expand_shape_method(scope)
            }

            #[allow(clippy::too_many_arguments)]
            fn __expand_is_in_bounds_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> NativeExpand<bool> {
                (**self).__expand_is_in_bounds_method(scope, pos)
            }
        }
    };
}

view_ops_read_ref!(&V);
view_ops_read_ref!(&mut V);

impl<T: CubePrimitive, C: Coordinates, V: ViewOperationsMut<T, C> + ?Sized> ViewOperationsMut<T, C>
    for &mut V
{
}
impl<T: CubePrimitive, C: Coordinates, V: ViewOperationsMutExpand<T, C> + ?Sized>
    ViewOperationsMutExpand<T, C> for &mut V
{
    #[allow(clippy::too_many_arguments)]
    fn __expand_write_method(&self, scope: &Scope, pos: C::ExpandType, value: T::ExpandType) {
        (**self).__expand_write_method(scope, pos, value);
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        value: T::ExpandType,
    ) {
        (**self).__expand_write_checked_method(scope, pos, value);
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> &mut SliceExpand<T> {
        (**self).__expand_as_linear_slice_mut_method(scope, pos, size)
    }

    #[allow(clippy::too_many_arguments)]
    fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &SliceExpand<T>,
        pos: C::ExpandType,
    ) {
        (**self).__expand_tensor_map_store_method(scope, shared_memory, pos);
    }
}

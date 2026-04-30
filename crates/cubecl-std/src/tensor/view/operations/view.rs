use super::*;
use crate::tensor::layout::Coordinates;
use crate::tensor::{View, ViewExpand};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

impl<T: CubePrimitive, C: Coordinates, IO: Clone> Vectorized for View<T, C, IO> {}
impl<T: CubePrimitive, C: Coordinates, IO: Clone> VectorizedExpand for ViewExpand<T, C, IO> {
    fn vector_size(&self) -> VectorSize {
        ViewExpand::vector_size(self)
    }
}

impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperations<T, C> for View<T, C, IO> {}
impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperationsExpand<T, C>
    for ViewExpand<T, C, IO>
{
    fn __expand_read_method(&self, scope: &Scope, pos: <C>::ExpandType) -> <T>::ExpandType {
        ViewExpand::__expand_read_method(self, scope, pos)
    }

    fn __expand_read_checked_method(&self, scope: &Scope, pos: <C>::ExpandType) -> <T>::ExpandType {
        ViewExpand::__expand_read_checked_method(self, scope, pos)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        mask_value: <T>::ExpandType,
    ) -> <T>::ExpandType {
        ViewExpand::__expand_read_masked_method(self, scope, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> <T>::ExpandType {
        ViewExpand::__expand_read_unchecked_method(self, scope, pos)
    }

    fn __expand_as_linear_slice_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> &SliceExpand<T> {
        ViewExpand::__expand_as_linear_slice_inner_method(self, scope, pos, end)
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <C>::ExpandType {
        ViewExpand::__expand_shape_method(self, scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<bool> {
        ViewExpand::__expand_is_in_bounds_method(self, scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        scope: &Scope,
        barrier: &NativeExpand<Barrier>,
        shared_memory: &mut SliceExpand<T>,
        pos: C::ExpandType,
    ) {
        ViewExpand::__expand_tensor_map_load_method(self, scope, barrier, shared_memory, pos)
    }
}

impl<T: CubePrimitive, C: Coordinates> ViewOperationsMut<T, C> for View<T, C, ReadWrite> {}
impl<T: CubePrimitive, C: Coordinates> ViewOperationsMutExpand<T, C>
    for ViewExpand<T, C, ReadWrite>
{
    fn __expand_write_method(&self, scope: &Scope, pos: <C>::ExpandType, value: <T>::ExpandType) {
        ViewExpand::__expand_write_method(self, scope, pos, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        ViewExpand::__expand_write_checked_method(self, scope, pos, value);
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> &mut SliceExpand<T> {
        ViewExpand::__expand_to_linear_slice_mut_inner_method(self, scope, pos, end)
    }

    fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &SliceExpand<T>,
        pos: C::ExpandType,
    ) {
        ViewExpand::__expand_tensor_map_store_method(self, scope, shared_memory, pos)
    }
}

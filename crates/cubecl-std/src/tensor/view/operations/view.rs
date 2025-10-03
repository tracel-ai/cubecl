use super::*;
use crate::tensor::layout::Coordinates;
use crate::tensor::{View, ViewExpand};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierExpand};

impl<T: CubePrimitive, C: Coordinates, IO: Clone> Lined for View<T, C, IO> {}
impl<T: CubePrimitive, C: Coordinates, IO: Clone> LinedExpand for ViewExpand<T, C, IO> {
    fn line_size(&self) -> u32 {
        ViewExpand::line_size(self)
    }
}

impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperations<T, C> for View<T, C, IO> {}
impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperationsExpand<T, C>
    for ViewExpand<T, C, IO>
{
    fn __expand_read_method(&self, scope: &mut Scope, pos: <C>::ExpandType) -> <T>::ExpandType {
        ViewExpand::__expand_read_method(self.clone(), scope, pos)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> <T>::ExpandType {
        ViewExpand::__expand_read_checked_method(self.clone(), scope, pos)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        mask_value: <T>::ExpandType,
    ) -> <T>::ExpandType {
        ViewExpand::__expand_read_masked_method(self.clone(), scope, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> <T>::ExpandType {
        ViewExpand::__expand_read_unchecked_method(self.clone(), scope, pos)
    }

    fn __expand_to_linear_slice_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> SliceExpand<T, ReadOnly> {
        ViewExpand::__expand_to_linear_slice_inner_method(self.clone(), scope, pos, end)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> <C>::ExpandType {
        ViewExpand::__expand_shape_method(self, scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<bool> {
        ViewExpand::__expand_is_in_bounds_method(self, scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        scope: &mut Scope,
        barrier: BarrierExpand,
        shared_memory: SliceExpand<T, ReadWrite>,
        pos: C::ExpandType,
    ) {
        ViewExpand::__expand_tensor_map_load_method(
            self.clone(),
            scope,
            barrier,
            shared_memory,
            pos,
        )
    }
}

impl<T: CubePrimitive, C: Coordinates> ViewOperationsMut<T, C> for View<T, C, ReadWrite> {}
impl<T: CubePrimitive, C: Coordinates> ViewOperationsMutExpand<T, C>
    for ViewExpand<T, C, ReadWrite>
{
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        ViewExpand::__expand_write_method(self.clone(), scope, pos, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        ViewExpand::__expand_write_checked_method(self.clone(), scope, pos, value);
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> SliceExpand<T, ReadWrite> {
        ViewExpand::__expand_to_linear_slice_mut_inner_method(self.clone(), scope, pos, end)
    }

    fn __expand_tensor_map_store_method(
        &self,
        scope: &mut Scope,
        shared_memory: SliceExpand<T, ReadOnly>,
        pos: C::ExpandType,
    ) {
        ViewExpand::__expand_tensor_map_store_method(self.clone(), scope, shared_memory, pos)
    }
}

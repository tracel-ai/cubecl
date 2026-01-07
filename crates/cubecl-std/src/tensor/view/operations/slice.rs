use super::*;
use crate::tensor::layout::Coords1d;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::BarrierExpand};

impl<T: CubePrimitive, IO: SliceVisibility> ViewOperations<T, Coords1d> for Slice<T, IO> {}
impl<T: CubePrimitive, IO: SliceVisibility> ViewOperationsExpand<T, Coords1d>
    for SliceExpand<T, IO>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <T>::ExpandType {
        <Self as ListExpand<T>>::__expand_read_method(self, scope, pos)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <T>::ExpandType {
        let len = self.__expand_len_method(scope);
        let in_bounds = lt::expand(scope, pos.clone(), len);
        let slice = self.clone().__expand_to_slice_method(scope);
        let zero = T::__expand_cast_from(scope, 0.into());
        read_masked::expand::<T>(scope, in_bounds, slice, pos, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        mask_value: <T>::ExpandType,
    ) -> <T>::ExpandType {
        let len = self.__expand_len_method(scope);
        let in_bounds = lt::expand(scope, pos.clone(), len);
        let slice = self.clone().__expand_to_slice_method(scope);
        read_masked::expand::<T>(scope, in_bounds, slice, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <T>::ExpandType {
        <Self as ListExpand<T>>::__expand_read_unchecked_method(self, scope, pos)
    }

    fn __expand_to_linear_slice_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<T, ReadOnly> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1usize.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = Min::__expand_min(scope, pos, end.clone());
        <Self as SliceOperatorExpand<T>>::__expand_slice_method(self, scope, start, end)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        <Self as ListExpand<T>>::__expand_len_method(self, scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<bool> {
        let len = self.__expand_shape_method(scope);
        lt::expand(scope, pos, len)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<T, ReadWrite>,
        _pos: ExpandElementTyped<usize>,
    ) {
        unimplemented!("Not a tensor map");
    }
}

impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for Slice<T, ReadWrite> {}
impl<T: CubePrimitive> ViewOperationsMutExpand<T, Coords1d> for SliceExpand<T, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        value: <T>::ExpandType,
    ) {
        <Self as ListMutExpand<T>>::__expand_write_method(self, scope, pos, value)
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        value: <T>::ExpandType,
    ) {
        let len = <Self as ListExpand<T>>::__expand_len_method(self, scope);
        let in_bounds = lt::expand(scope, pos.clone(), len);
        if_expand(scope, in_bounds.into(), |scope| {
            <Self as ListMutExpand<T>>::__expand_write_method(self, scope, pos, value)
        })
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<T, ReadWrite> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1usize.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = Min::__expand_min(scope, pos, end.clone());
        <Self as SliceMutOperatorExpand<T>>::__expand_slice_mut_method(self, scope, start, end)
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<T, ReadOnly>,
        _pos: <Coords1d as CubeType>::ExpandType,
    ) {
        unimplemented!("Not a tensor map");
    }
}

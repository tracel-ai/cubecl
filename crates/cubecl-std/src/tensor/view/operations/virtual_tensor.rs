use super::*;
use crate::tensor::{
    layout::Coords1d,
    r#virtual::{VirtualTensor, VirtualTensorExpand},
};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::BarrierExpand};

impl<T: Numeric, IO: Clone> ViewOperations<Line<T>, Coords1d> for VirtualTensor<T, IO> {}
impl<T: Numeric, IO: Clone> ViewOperationsExpand<Line<T>, Coords1d> for VirtualTensorExpand<T, IO> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<T> as CubeType>::ExpandType {
        <Self as ListExpand<Line<T>>>::__expand_read_method(self, scope, pos)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<T> as CubeType>::ExpandType {
        let zero = Line::__expand_cast_from(scope, 0.into());
        self.__expand_read_masked_method(scope, pos, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        mask_value: <Line<T> as CubeType>::ExpandType,
    ) -> <Line<T> as CubeType>::ExpandType {
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos.clone());
        let slice = self.clone().__expand_to_slice_method(scope);
        read_masked::expand::<Line<T>>(scope, in_bounds, slice, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> <Line<T> as CubeType>::ExpandType {
        <Self as ListExpand<Line<T>>>::__expand_read_unchecked_method(self, scope, pos)
    }

    fn __expand_to_linear_slice_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<T>, ReadOnly> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1u32.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = Min::__expand_min(scope, pos, end.clone());
        <Self as SliceOperatorExpand<Line<T>>>::__expand_slice_method(self, scope, start, end)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        self.clone().__expand_buffer_len_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<bool> {
        let len = self.clone().__expand_buffer_len_method(scope);
        lt::expand(scope, pos, len)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<Line<T>, ReadWrite>,
        _pos: ExpandElementTyped<u32>,
    ) {
        unimplemented!("Not a tensor map");
    }
}

impl<T: Numeric> ViewOperationsMut<Line<T>, Coords1d> for VirtualTensor<T, ReadWrite> {}
impl<T: Numeric> ViewOperationsMutExpand<Line<T>, Coords1d> for VirtualTensorExpand<T, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        value: <Line<T> as CubeType>::ExpandType,
    ) {
        <Self as ListMutExpand<Line<T>>>::__expand_write_method(self, scope, pos, value)
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        value: <Line<T> as CubeType>::ExpandType,
    ) {
        let len = self.clone().__expand_buffer_len_method(scope);
        let in_bounds = lt::expand(scope, pos.clone(), len);
        if_expand(scope, in_bounds.into(), |scope| {
            <Self as ListMutExpand<Line<T>>>::__expand_write_method(self, scope, pos, value)
        })
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<T>, ReadWrite> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1u32.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = Min::__expand_min(scope, pos, end.clone());
        <Self as SliceMutOperatorExpand<Line<T>>>::__expand_slice_mut_method(
            self, scope, start, end,
        )
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<Line<T>, ReadOnly>,
        _pos: <Coords1d as CubeType>::ExpandType,
    ) {
        unimplemented!("Not a tensor map");
    }
}

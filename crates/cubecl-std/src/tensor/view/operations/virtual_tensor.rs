use super::*;
use crate::tensor::{
    layout::Coords1d,
    r#virtual::{VirtualTensor, VirtualTensorExpand},
};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::BarrierExpand};

impl<T: Numeric, N: Size, IO: Clone> ViewOperations<Line<T, N>, Coords1d>
    for VirtualTensor<T, N, IO>
{
}
impl<T: Numeric, N: Size, IO: Clone> ViewOperationsExpand<Line<T, N>, Coords1d>
    for VirtualTensorExpand<T, N, IO>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <Line<T, N> as CubeType>::ExpandType {
        <Self as ListExpand<Line<T, N>>>::__expand_read_method(self, scope, pos)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <Line<T, N> as CubeType>::ExpandType {
        let zero = Line::__expand_cast_from(scope, 0.into());
        self.__expand_read_masked_method(scope, pos, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        mask_value: <Line<T, N> as CubeType>::ExpandType,
    ) -> <Line<T, N> as CubeType>::ExpandType {
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos.clone());
        let slice = self.clone().__expand_to_slice_method(scope);
        read_masked::expand::<Line<T, N>>(scope, in_bounds, slice, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> <Line<T, N> as CubeType>::ExpandType {
        <Self as ListExpand<Line<T, N>>>::__expand_read_unchecked_method(self, scope, pos)
    }

    fn __expand_to_linear_slice_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Line<T, N>, ReadOnly> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1usize.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end.clone());
        <Self as SliceOperatorExpand<Line<T, N>>>::__expand_slice_method(self, scope, start, end)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        self.clone().__expand_buffer_len_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<bool> {
        let len = self.clone().__expand_buffer_len_method(scope);
        lt::expand(scope, pos, len)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<Line<T, N>, ReadWrite>,
        _pos: ExpandElementTyped<usize>,
    ) {
        unimplemented!("Not a tensor map");
    }
}

impl<T: Numeric, N: Size> ViewOperationsMut<Line<T, N>, Coords1d>
    for VirtualTensor<T, N, ReadWrite>
{
}
impl<T: Numeric, N: Size> ViewOperationsMutExpand<Line<T, N>, Coords1d>
    for VirtualTensorExpand<T, N, ReadWrite>
{
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        value: <Line<T, N> as CubeType>::ExpandType,
    ) {
        <Self as ListMutExpand<Line<T, N>>>::__expand_write_method(self, scope, pos, value)
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        value: <Line<T, N> as CubeType>::ExpandType,
    ) {
        let len = self.clone().__expand_buffer_len_method(scope);
        let in_bounds = lt::expand(scope, pos.clone(), len);
        if_expand(scope, in_bounds.into(), |scope| {
            <Self as ListMutExpand<Line<T, N>>>::__expand_write_method(self, scope, pos, value)
        })
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        scope: &mut Scope,
        pos: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Line<T, N>, ReadWrite> {
        // Convert to exclusive end
        let end = add::expand(scope, end, 1usize.into());
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end.clone());
        <Self as SliceMutOperatorExpand<Line<T, N>>>::__expand_slice_mut_method(
            self, scope, start, end,
        )
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<Line<T, N>, ReadOnly>,
        _pos: <Coords1d as CubeType>::ExpandType,
    ) {
        unimplemented!("Not a tensor map");
    }
}

use super::*;
use crate::tensor::{
    layout::Coords1d,
    r#virtual::{VirtualTensor, VirtualTensorExpand},
};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::Barrier};

impl<T: Numeric, N: Size, IO: Clone> ViewOperations<Vector<T, N>, Coords1d>
    for VirtualTensor<T, N, IO>
{
}
impl<T: Numeric, N: Size, IO: Clone> ViewOperationsExpand<Vector<T, N>, Coords1d>
    for VirtualTensorExpand<T, N, IO>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<T, N> as CubeType>::ExpandType {
        Self::__expand_read_method(self, scope, pos)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<T, N> as CubeType>::ExpandType {
        let zero = Vector::__expand_cast_from(scope, 0.into());
        self.__expand_read_masked_method(scope, pos, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        mask_value: <Vector<T, N> as CubeType>::ExpandType,
    ) -> <Vector<T, N> as CubeType>::ExpandType {
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let slice = self.__expand_as_slice_method(scope);
        read_masked::expand::<Vector<T, N>>(scope, in_bounds, slice, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <Vector<T, N> as CubeType>::ExpandType {
        self.__expand_read_method(scope, pos)
    }

    fn __expand_as_linear_slice_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<T, N>> {
        // Convert to exclusive end
        let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end);
        <Self as SliceOperatorExpand<Vector<T, N>>>::__expand_slice_method(self, scope, start, end)
    }

    fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.clone().__expand_buffer_len_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<bool> {
        let len = self.clone().__expand_buffer_len_method(scope);
        pos.__expand_lt_method(scope, &len)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<T, N>>,
        _pos: NativeExpand<usize>,
    ) {
        unimplemented!("Not a tensor map");
    }
}

impl<T: Numeric, N: Size> ViewOperationsMut<Vector<T, N>, Coords1d>
    for VirtualTensor<T, N, ReadWrite>
{
}
impl<T: Numeric, N: Size> ViewOperationsMutExpand<Vector<T, N>, Coords1d>
    for VirtualTensorExpand<T, N, ReadWrite>
{
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <Vector<T, N> as CubeType>::ExpandType,
    ) {
        self.__expand_write_method(scope, pos, value)
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <Vector<T, N> as CubeType>::ExpandType,
    ) {
        let len = self.clone().__expand_buffer_len_method(scope);
        let in_bounds = pos.__expand_lt_method(scope, &len);
        if_expand(scope, in_bounds, |scope| {
            self.__expand_write_method(scope, pos, value)
        })
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<Vector<T, N>> {
        // Convert to exclusive end
        let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end);
        let mut this = self.clone();
        let slice = <Self as SliceOperatorExpand<Vector<T, N>>>::__expand_slice_mut_method(
            &mut this, scope, start, end,
        );
        // Arrays are internally references, so this is actually 'a
        unsafe { core::mem::transmute(slice) }
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<Vector<T, N>>,
        _pos: <Coords1d as CubeType>::ExpandType,
    ) {
        unimplemented!("Not a tensor map");
    }
}

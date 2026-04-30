use super::*;
use crate::tensor::layout::Coords1d;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::Barrier};

impl<T: CubePrimitive> ViewOperations<T, Coords1d> for Box<[T]> {}
impl<T: CubePrimitive> ViewOperations<T, Coords1d> for [T] {}
impl<T: CubePrimitive> ViewOperationsExpand<T, Coords1d> for SliceExpand<T> {
    fn __expand_read_method(&self, scope: &Scope, pos: NativeExpand<usize>) -> <T>::ExpandType {
        <Self as ListExpand<T>>::__expand_read_method(self, scope, pos).__expand_deref_method(scope)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <T>::ExpandType {
        let len = self.__expand_len_method(scope);
        let in_bounds = pos.__expand_lt_method(scope, &len);
        let zero = T::__expand_cast_from(scope, 0.into());
        read_masked::expand::<T>(scope, in_bounds, self, pos, zero)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        mask_value: <T>::ExpandType,
    ) -> <T>::ExpandType {
        let len = self.__expand_len_method(scope);
        let in_bounds = pos.__expand_lt_method(scope, &len);
        read_masked::expand::<T>(scope, in_bounds, self, pos, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> <T>::ExpandType {
        <Self as ListExpand<T>>::__expand_read_unchecked_method(self, scope, pos)
            .__expand_deref_method(scope)
    }

    fn __expand_as_linear_slice_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<T> {
        // Convert to exclusive end
        let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end);
        <Self as SliceOperatorExpand<T>>::__expand_slice_method(self, scope, start, end)
    }

    fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
        <Self as ListExpand<T>>::__expand_len_method(self, scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
    ) -> NativeExpand<bool> {
        let len = self.__expand_shape_method(scope);
        pos.__expand_lt_method(scope, &len)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<T>,
        _pos: NativeExpand<usize>,
    ) {
        unimplemented!("Not a tensor map");
    }
}

impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for Box<[T]> {}
impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for [T] {}
impl<T: CubePrimitive> ViewOperationsMutExpand<T, Coords1d> for SliceExpand<T> {
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <T>::ExpandType,
    ) {
        let mut this = *self;
        <Self as ListExpand<T>>::__expand_write_method(&mut this, scope, pos)
            .__expand_assign_method(scope, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        value: <T>::ExpandType,
    ) {
        let mut this = *self;
        let len = <Self as ListExpand<T>>::__expand_len_method(self, scope);
        let in_bounds = pos.__expand_lt_method(scope, &len);
        if_expand(scope, in_bounds, |scope| {
            <Self as ListExpand<T>>::__expand_write_method(&mut this, scope, pos)
                .__expand_assign_method(scope, value)
        })
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &mut SliceExpand<T> {
        // Convert to exclusive end
        let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
        // Handling for shapes that are 0 in at least one dim, ensures the slice is not
        // negative length.
        let start = clamp_max::expand(scope, pos, end);
        let mut this = *self;
        let slice = <Self as SliceOperatorExpand<T>>::__expand_slice_mut_method(
            &mut this, scope, start, end,
        );
        // Slices are internally references, so this is actually 'a
        unsafe { core::mem::transmute(slice) }
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &Scope,
        _shared_memory: &SliceExpand<T>,
        _pos: <Coords1d as CubeType>::ExpandType,
    ) {
        unimplemented!("Not a tensor map");
    }
}

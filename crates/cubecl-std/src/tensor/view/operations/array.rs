use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::Barrier};

use crate::tensor::{
    ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand,
    layout::Coords1d,
};

macro_rules! impl_operations_1d {
    ($ty: ty, $expand: ty) => {
        impl<T: CubePrimitive> ViewOperations<T, Coords1d> for $ty {}
        impl<T: CubePrimitive> ViewOperationsExpand<T, Coords1d> for $expand {
            fn __expand_read_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
            ) -> <T>::ExpandType {
                <Self as ListExpand<T>>::__expand_read_method(&self, scope, pos)
                    .__expand_deref_method(scope)
            }

            fn __expand_read_checked_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
            ) -> <T>::ExpandType {
                let len = self.__expand_buffer_len_method(scope);
                let in_bounds = pos.__expand_lt_method(scope, &len);
                let slice = self.__expand_to_slice_method(scope);
                let zero = T::__expand_cast_from(scope, 0.into());
                read_masked::expand::<T>(scope, in_bounds, slice, pos, zero)
            }

            fn __expand_read_masked_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
                mask_value: <T>::ExpandType,
            ) -> <T>::ExpandType {
                let len = self.__expand_buffer_len_method(scope);
                let in_bounds = pos.__expand_lt_method(scope, &len);
                let slice = self.__expand_to_slice_method(scope);
                read_masked::expand::<T>(scope, in_bounds, slice, pos, mask_value)
            }

            fn __expand_read_unchecked_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
            ) -> <T>::ExpandType {
                <Self as ListExpand<T>>::__expand_read_unchecked_method(self, scope, pos)
                    .__expand_deref_method(scope)
            }

            fn __expand_to_linear_slice_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
                end: NativeExpand<usize>,
            ) -> &SliceExpand<T, ReadOnly> {
                // Convert to exclusive end
                let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
                // Handling for shapes that are 0 in at least one dim, ensures the slice is not
                // negative length.
                let start = clamp_max::expand(scope, pos, end);
                <Self as SliceOperatorExpand<T>>::__expand_slice_method(self, scope, start, end)
            }

            fn __expand_shape_method(&self, scope: &Scope) -> NativeExpand<usize> {
                self.__expand_buffer_len_method(scope)
            }

            fn __expand_is_in_bounds_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
            ) -> NativeExpand<bool> {
                let len = self.__expand_buffer_len_method(scope);
                pos.__expand_lt_method(scope, &len)
            }

            fn __expand_tensor_map_load_method(
                &self,
                _scope: &Scope,
                _barrier: &NativeExpand<Barrier>,
                _shared_memory: &mut SliceExpand<T, ReadWrite>,
                _pos: NativeExpand<usize>,
            ) {
                unimplemented!("Not a tensor map");
            }
        }

        impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for $ty {}
        impl<T: CubePrimitive> ViewOperationsMutExpand<T, Coords1d> for $expand {
            fn __expand_write_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
                value: <T>::ExpandType,
            ) {
                <Self as ListMutExpand<T>>::__expand_write_method(&self, scope, pos)
                    .__expand_assign_method(scope, value);
            }

            fn __expand_write_checked_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
                value: <T>::ExpandType,
            ) {
                let len = self.clone().__expand_buffer_len_method(scope);
                let in_bounds = pos.__expand_lt_method(scope, &len);
                if_expand(scope, in_bounds.into(), |scope| {
                    <Self as ListMutExpand<T>>::__expand_write_method(&self, scope, pos)
                        .__expand_assign_method(scope, value);
                })
            }

            fn __expand_to_linear_slice_mut_method(
                &self,
                scope: &Scope,
                pos: NativeExpand<usize>,
                end: NativeExpand<usize>,
            ) -> &mut SliceExpand<T, ReadWrite> {
                // Convert to exclusive end
                let end = end.__expand_add_method(scope, 1usize.into_expand(scope));
                // Handling for shapes that are 0 in at least one dim, ensures the slice is not
                // negative length.
                let start = clamp_max::expand(scope, pos, end.clone());
                let mut this = self.clone();
                let slice = <Self as SliceMutOperatorExpand<T>>::__expand_slice_mut_method(
                    &mut this, scope, start, end,
                );
                // Arrays are internally references, so this is actually 'a
                unsafe { core::mem::transmute(slice) }
            }

            fn __expand_tensor_map_store_method(
                &self,
                _scope: &Scope,
                _shared_memory: &SliceExpand<T, ReadOnly>,
                _pos: <Coords1d as CubeType>::ExpandType,
            ) {
                unimplemented!("Not a tensor map");
            }
        }
    };
}

impl_operations_1d!(Array<T>, NativeExpand<Array<T>>);
impl_operations_1d!(Tensor<T>, NativeExpand<Tensor<T>>);
impl_operations_1d!(SharedMemory<T>, NativeExpand<SharedMemory<T>>);

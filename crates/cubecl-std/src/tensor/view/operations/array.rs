use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, prelude::barrier::BarrierExpand};

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
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
            ) -> <T>::ExpandType {
                <Self as ListExpand<T>>::__expand_read_method(&self, scope, pos)
            }

            fn __expand_read_checked_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
            ) -> <T>::ExpandType {
                let len = self.clone().__expand_buffer_len_method(scope);
                let in_bounds = lt::expand(scope, pos.clone(), len);
                let slice = self.clone().__expand_to_slice_method(scope);
                let zero = T::__expand_cast_from(scope, 0.into());
                read_masked::expand::<T>(scope, in_bounds, slice, pos, zero)
            }

            fn __expand_read_masked_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
                mask_value: <T>::ExpandType,
            ) -> <T>::ExpandType {
                let len = self.clone().__expand_buffer_len_method(scope);
                let in_bounds = lt::expand(scope, pos.clone(), len);
                let slice = self.clone().__expand_to_slice_method(scope);
                read_masked::expand::<T>(scope, in_bounds, slice, pos, mask_value)
            }

            fn __expand_read_unchecked_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
            ) -> <T>::ExpandType {
                <Self as ListExpand<T>>::__expand_read_unchecked_method(self, scope, pos)
            }

            fn __expand_to_linear_slice_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
                end: ExpandElementTyped<u32>,
            ) -> SliceExpand<T, ReadOnly> {
                // Convert to exclusive end
                let end = add::expand(scope, end, 1u32.into());
                // Handling for shapes that are 0 in at least one dim, ensures the slice is not
                // negative length.
                let start = Min::__expand_min(scope, pos, end.clone());
                <Self as SliceOperatorExpand<T>>::__expand_slice_method(self, scope, start, end)
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
                _shared_memory: SliceExpand<T, ReadWrite>,
                _pos: ExpandElementTyped<u32>,
            ) {
                unimplemented!("Not a tensor map");
            }
        }

        impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for $ty {}
        impl<T: CubePrimitive> ViewOperationsMutExpand<T, Coords1d> for $expand {
            fn __expand_write_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
                value: <T>::ExpandType,
            ) {
                <Self as ListMutExpand<T>>::__expand_write_method(&self, scope, pos, value)
            }

            fn __expand_write_checked_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
                value: <T>::ExpandType,
            ) {
                let len = self.clone().__expand_buffer_len_method(scope);
                let in_bounds = lt::expand(scope, pos.clone(), len);
                if_expand(scope, in_bounds.into(), |scope| {
                    <Self as ListMutExpand<T>>::__expand_write_method(&self, scope, pos, value)
                })
            }

            fn __expand_to_linear_slice_mut_method(
                &self,
                scope: &mut Scope,
                pos: ExpandElementTyped<u32>,
                end: ExpandElementTyped<u32>,
            ) -> SliceExpand<T, ReadWrite> {
                // Convert to exclusive end
                let end = add::expand(scope, end, 1u32.into());
                // Handling for shapes that are 0 in at least one dim, ensures the slice is not
                // negative length.
                let start = Min::__expand_min(scope, pos, end.clone());
                <Self as SliceMutOperatorExpand<T>>::__expand_slice_mut_method(
                    self, scope, start, end,
                )
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
    };
}

impl_operations_1d!(Array<T>, ExpandElementTyped<Array<T>>);
impl_operations_1d!(Tensor<T>, ExpandElementTyped<Tensor<T>>);
impl_operations_1d!(SharedMemory<T>, ExpandElementTyped<SharedMemory<T>>);

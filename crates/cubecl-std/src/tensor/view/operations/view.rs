use super::*;
use crate::tensor::{View, ViewExpand};
use crate::tensor::{ViewMut, ViewMutExpand, layout::Coordinates};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

macro_rules! impl_base {
    ($ty: ident, $expand: ident) => {
        impl<'a, T: CubePrimitive, C: Coordinates + 'a> Vectorized for $ty<'a, T, C> {}
        impl<'a, T: CubePrimitive, C: Coordinates + 'a> VectorizedExpand for $expand<'a, T, C> {
            fn vector_size(&self) -> VectorSize {
                self.inner.vector_size()
            }
        }

        impl<'a, T: CubePrimitive, C: Coordinates + 'a> ViewOperations<T, C> for $ty<'a, T, C> {}
        impl<'a, T: CubePrimitive, C: Coordinates + 'a> ViewOperationsExpand<T, C>
            for $expand<'a, T, C>
        {
            fn __expand_read_method(&self, scope: &Scope, pos: <C>::ExpandType) -> <T>::ExpandType {
                $expand::__expand_read_method(self, scope, pos)
            }

            fn __expand_read_checked_method(
                &self,
                scope: &Scope,
                pos: <C>::ExpandType,
            ) -> <T>::ExpandType {
                $expand::__expand_read_checked_method(self, scope, pos)
            }

            fn __expand_read_masked_method(
                &self,
                scope: &Scope,
                pos: <C>::ExpandType,
                mask_value: <T>::ExpandType,
            ) -> <T>::ExpandType {
                $expand::__expand_read_masked_method(self, scope, pos, mask_value)
            }

            fn __expand_read_unchecked_method(
                &self,
                scope: &Scope,
                pos: <C>::ExpandType,
            ) -> <T>::ExpandType {
                $expand::__expand_read_unchecked_method(self, scope, pos)
            }

            fn __expand_as_linear_slice_method(
                &self,
                scope: &Scope,
                pos: <C>::ExpandType,
                end: <C>::ExpandType,
            ) -> &SliceExpand<T> {
                $expand::__expand_as_linear_slice_inner_method(self, scope, pos, end)
            }

            fn __expand_shape_method(&self, scope: &Scope) -> <C>::ExpandType {
                $expand::__expand_shape_method(self, scope)
            }

            fn __expand_is_in_bounds_method(
                &self,
                scope: &Scope,
                pos: <C>::ExpandType,
            ) -> NativeExpand<bool> {
                $expand::__expand_is_in_bounds_method(self, scope, pos)
            }

            fn __expand_tensor_map_load_method(
                &self,
                scope: &Scope,
                barrier: &NativeExpand<Barrier>,
                shared_memory: &mut SliceExpand<T>,
                pos: C::ExpandType,
            ) {
                $expand::__expand_tensor_map_load_method(self, scope, barrier, shared_memory, pos)
            }
        }
    };
}

impl_base!(View, ViewExpand);
impl_base!(ViewMut, ViewMutExpand);

impl<'a, T: CubePrimitive, C: Coordinates + 'a> ViewOperationsMut<T, C> for ViewMut<'a, T, C> {}
impl<'a, T: CubePrimitive, C: Coordinates + 'a> ViewOperationsMutExpand<T, C>
    for ViewMutExpand<'a, T, C>
{
    fn __expand_write_method(&self, scope: &Scope, pos: <C>::ExpandType, value: <T>::ExpandType) {
        self.inner.__expand_write_method(scope, pos, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        self.inner.__expand_write_checked_method(scope, pos, value);
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> &mut SliceExpand<T> {
        self.inner
            .__expand_as_linear_slice_mut_method(scope, pos, end)
    }

    fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &SliceExpand<T>,
        pos: C::ExpandType,
    ) {
        self.inner
            .__expand_tensor_map_store_method(scope, shared_memory, pos)
    }
}

use super::*;
use crate::tensor::layout::*;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierExpand};

// We don't know the linear layout, so only implement TMA loads/stores
macro_rules! impl_tensor_map {
    ($dim: literal, $coords: ty, $($var: ident),*) => {
        paste::paste! {
            impl<T: CubePrimitive> ViewOperations<T, $coords> for TensorMap<T> {}
            impl<T: CubePrimitive> ViewOperationsExpand<T, $coords> for ExpandElementTyped<TensorMap<T>> {
                fn __expand_read_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                ) -> <T as CubeType>::ExpandType {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_read_checked_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                ) -> <T as CubeType>::ExpandType {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_read_masked_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                    _mask_value: <T as CubeType>::ExpandType,
                ) -> <T as CubeType>::ExpandType {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_read_unchecked_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                ) -> <T as CubeType>::ExpandType {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_to_linear_slice_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                    _end: <$coords as CubeType>::ExpandType,
                ) -> SliceExpand<T, ReadOnly> {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_shape_method(&self, _scope: &mut Scope) -> <$coords as CubeType>::ExpandType {
                    unimplemented!("Can't read from tensor map");
                }

                fn __expand_is_in_bounds_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                ) -> ExpandElementTyped<bool> {
                    unimplemented!("Can't read from tensor map");
                }

                #[allow(unused_parens)]
                fn __expand_tensor_map_load_method(
                    &self,
                    scope: &mut Scope,
                    barrier: BarrierExpand,
                    shared_memory: SliceExpand<T, ReadWrite>,
                    pos: <$coords as CubeType>::ExpandType,
                ) {
                    let shared = shared_memory.__expand_try_cast_unchecked_method(scope);
                    let ($($var),*) = pos;
                    let ($($var),*) = ($(i32::__expand_cast_from(scope, $var)),*);
                    barrier.[<__expand_tma_load_ $dim d_method>]::<T>(scope, self.clone(), shared, $($var),*);
                }
            }

            impl<T: CubePrimitive> ViewOperationsMut<T, $coords> for TensorMap<T> {}
            impl<T: CubePrimitive> ViewOperationsMutExpand<T, $coords> for ExpandElementTyped<TensorMap<T>> {
                fn __expand_write_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                    _value: <T as CubeType>::ExpandType,
                ) {
                    unimplemented!("Can't write to tensor map");
                }

                fn __expand_write_checked_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                    _value: <T as CubeType>::ExpandType,
                ) {
                    unimplemented!("Can't write to tensor map");
                }

                fn __expand_to_linear_slice_mut_method(
                    &self,
                    _scope: &mut Scope,
                    _pos: <$coords as CubeType>::ExpandType,
                    _end: <$coords as CubeType>::ExpandType,
                ) -> SliceExpand<T, ReadWrite> {
                    unimplemented!("Can't write to tensor map");
                }

                #[allow(unused_parens)]
                fn __expand_tensor_map_store_method(
                    &self,
                    scope: &mut Scope,
                    shared_memory: SliceExpand<T, ReadOnly>,
                    pos: <$coords as CubeType>::ExpandType,
                ) {
                    let shared = shared_memory.__expand_try_cast_unchecked_method(scope);
                    let ($($var),*) = pos;
                    let ($($var),*) = ($(i32::__expand_cast_from(scope, $var)),*);
                    [<tma_store_ $dim d>]::expand(scope, shared, self.clone(), $($var),*);
                }
            }
        }
    };
}

impl_tensor_map!(1, Coords1d, x);
impl_tensor_map!(2, Coords2d, x, y);
impl_tensor_map!(3, Coords3d, x, y, z);
impl_tensor_map!(4, Coords4d, x, y, z, v);
impl_tensor_map!(5, Coords5d, x, y, z, v, w);

impl_tensor_map!(1, Coords1i, x);
impl_tensor_map!(2, Coords2i, x, y);
impl_tensor_map!(3, Coords3i, x, y, z);
impl_tensor_map!(4, Coords4i, x, y, z, v);
impl_tensor_map!(5, Coords5i, x, y, z, v, w);

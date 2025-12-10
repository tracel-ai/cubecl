use super::*;
use crate::tensor::layout::*;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierExpand};

// We don't know the linear layout, so only implement TMA loads/stores
macro_rules! impl_tensor_map {
    ($dim: literal, $coords: ty, $($var: ident),*) => {
        paste::paste! {
            impl<T: CubePrimitive> ViewOperations<T, $coords> for TensorMap<T, Tiled> {}
            impl<T: CubePrimitive> ViewOperationsExpand<T, $coords> for ExpandElementTyped<TensorMap<T, Tiled>> {
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
                    // Bounds checks are done in hardware, so treat them as always in bounds for the kernels
                    true.into()
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

            impl<T: CubePrimitive> ViewOperationsMut<T, $coords> for TensorMap<T, Tiled> {}
            impl<T: CubePrimitive> ViewOperationsMutExpand<T, $coords> for ExpandElementTyped<TensorMap<T, Tiled>> {
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

// We don't know the linear layout, so only implement TMA loads
macro_rules! impl_tensor_map_im2col {
    ($dim: literal, $coords: ty, $($pos: ident),*; $($offs: ident),*) => {
        paste::paste! {
            impl<T: CubePrimitive> ViewOperations<T, $coords> for TensorMap<T, Im2col> {}
            impl<T: CubePrimitive> ViewOperationsExpand<T, $coords> for ExpandElementTyped<TensorMap<T, Im2col>> {
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
                    // Bounds checks are done in hardware, so treat them as always in bounds for the kernels
                    true.into()
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
                    let ($($pos),*) = pos.0;
                    let ($($pos),*) = ($(i32::__expand_cast_from(scope, $pos)),*);
                    let ($($offs),*) = pos.1;
                    let ($($offs),*) = ($(u16::__expand_cast_from(scope, $offs)),*);

                    barrier.[<__expand_tma_load_im2col_ $dim d_method>]::<T>(scope, self.clone(), shared, $($pos),*, $($offs),*);
                }
            }
        }
    };
}

impl_tensor_map_im2col!(3, (Coords3d, Coords1d), n, w, c; x);
impl_tensor_map_im2col!(4, (Coords4d, Coords2d), n, h, w, c; y, x);
impl_tensor_map_im2col!(5, (Coords5d, Coords3d), n, d, h, w, c; z, y, x);

impl_tensor_map_im2col!(3, (Coords3i, Coords1d), n, w, c; x);
impl_tensor_map_im2col!(4, (Coords4i, Coords2d), n, h, w, c; y, x);
impl_tensor_map_im2col!(5, (Coords5i, Coords3d), n, d, h, w, c; z, y, x);

fn as_i32<T: CubePrimitive>(
    scope: &mut Scope,
    pos: &SequenceExpand<T>,
    i: u32,
) -> ExpandElementTyped<i32> {
    let x = pos.__expand_index_method(scope, i.into());
    i32::__expand_cast_from(scope, x)
}

fn as_u16<T: CubePrimitive>(
    scope: &mut Scope,
    offs: &SequenceExpand<T>,
    i: u32,
) -> ExpandElementTyped<u16> {
    let x = offs.__expand_index_method(scope, i.into());
    u16::__expand_cast_from(scope, x)
}

impl<T: CubePrimitive, N: CubePrimitive + Coordinates> ViewOperations<T, Sequence<N>>
    for TensorMap<T, Tiled>
{
}
impl<T: CubePrimitive, N: CubePrimitive + Coordinates> ViewOperationsExpand<T, Sequence<N>>
    for ExpandElementTyped<TensorMap<T, Tiled>>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_masked_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
        _mask_value: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
        _end: SequenceExpand<N>,
    ) -> SliceExpand<T, ReadOnly> {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_shape_method(&self, _scope: &mut Scope) -> SequenceExpand<N> {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_is_in_bounds_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
    ) -> ExpandElementTyped<bool> {
        // Bounds checks are done in hardware, so treat them as always in bounds for the kernels
        true.into()
    }

    #[allow(unused_parens)]
    fn __expand_tensor_map_load_method(
        &self,
        scope: &mut Scope,
        barrier: BarrierExpand,
        shared_memory: SliceExpand<T, ReadWrite>,
        pos: SequenceExpand<N>,
    ) {
        let shared = shared_memory.__expand_try_cast_unchecked_method(scope);
        let rank = pos.len();
        let pos = &pos;
        match rank {
            1 => {
                let x = as_i32(scope, pos, 0);
                barrier.__expand_tma_load_1d_method(scope, self.clone(), shared, x);
            }
            2 => {
                let y = as_i32(scope, pos, 0);
                let x = as_i32(scope, pos, 1);
                barrier.__expand_tma_load_2d_method(scope, self.clone(), shared, y, x);
            }
            3 => {
                let z = as_i32(scope, pos, 0);
                let y = as_i32(scope, pos, 1);
                let x = as_i32(scope, pos, 2);
                barrier.__expand_tma_load_3d_method(scope, self.clone(), shared, z, y, x);
            }
            4 => {
                let w = as_i32(scope, pos, 0);
                let z = as_i32(scope, pos, 1);
                let y = as_i32(scope, pos, 2);
                let x = as_i32(scope, pos, 3);
                barrier.__expand_tma_load_4d_method(scope, self.clone(), shared, w, z, y, x);
            }
            5 => {
                let v = as_i32(scope, pos, 0);
                let w = as_i32(scope, pos, 1);
                let z = as_i32(scope, pos, 2);
                let y = as_i32(scope, pos, 3);
                let x = as_i32(scope, pos, 4);
                barrier.__expand_tma_load_5d_method(scope, self.clone(), shared, v, w, z, y, x);
            }
            _ => panic!("TMA only supports 1D-5D loads"),
        }
    }
}

impl<T: CubePrimitive, N: CubePrimitive + Coordinates> ViewOperationsMut<T, Sequence<N>>
    for TensorMap<T, Tiled>
{
}
impl<T: CubePrimitive, N: CubePrimitive + Coordinates> ViewOperationsMutExpand<T, Sequence<N>>
    for ExpandElementTyped<TensorMap<T, Tiled>>
{
    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
        _value: <T as CubeType>::ExpandType,
    ) {
        unimplemented!("Can't write to tensor map");
    }

    fn __expand_write_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
        _value: <T as CubeType>::ExpandType,
    ) {
        unimplemented!("Can't write to tensor map");
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        _scope: &mut Scope,
        _pos: SequenceExpand<N>,
        _end: SequenceExpand<N>,
    ) -> SliceExpand<T, ReadWrite> {
        unimplemented!("Can't write to tensor map");
    }

    #[allow(unused_parens)]
    fn __expand_tensor_map_store_method(
        &self,
        scope: &mut Scope,
        shared_memory: SliceExpand<T, ReadOnly>,
        pos: SequenceExpand<N>,
    ) {
        let shared = shared_memory.__expand_try_cast_unchecked_method(scope);
        let rank = pos.len();
        let pos = &pos;
        match rank {
            1 => {
                let x = as_i32(scope, pos, 0);
                tma_store_1d::expand(scope, shared, self.clone(), x);
            }
            2 => {
                let y = as_i32(scope, pos, 0);
                let x = as_i32(scope, pos, 1);
                tma_store_2d::expand(scope, shared, self.clone(), y, x);
            }
            3 => {
                let z = as_i32(scope, pos, 0);
                let y = as_i32(scope, pos, 1);
                let x = as_i32(scope, pos, 2);
                tma_store_3d::expand(scope, shared, self.clone(), z, y, x);
            }
            4 => {
                let w = as_i32(scope, pos, 0);
                let z = as_i32(scope, pos, 1);
                let y = as_i32(scope, pos, 2);
                let x = as_i32(scope, pos, 3);
                tma_store_4d::expand(scope, shared, self.clone(), w, z, y, x);
            }
            5 => {
                let v = as_i32(scope, pos, 0);
                let w = as_i32(scope, pos, 1);
                let z = as_i32(scope, pos, 2);
                let y = as_i32(scope, pos, 3);
                let x = as_i32(scope, pos, 4);
                tma_store_5d::expand(scope, shared, self.clone(), v, w, z, y, x);
            }
            _ => panic!("TMA store supports 1D-5D loads"),
        }
    }
}

impl<T: CubePrimitive, P: CubePrimitive + Coordinates, O: CubePrimitive + Coordinates>
    ViewOperations<T, (Sequence<P>, Sequence<O>)> for TensorMap<T, Im2col>
{
}
impl<T: CubePrimitive, P: CubePrimitive + Coordinates, O: CubePrimitive + Coordinates>
    ViewOperationsExpand<T, (Sequence<P>, Sequence<O>)>
    for ExpandElementTyped<TensorMap<T, Im2col>>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_masked_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
        _mask_value: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
    ) -> <T as CubeType>::ExpandType {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
        _end: (SequenceExpand<P>, SequenceExpand<O>),
    ) -> SliceExpand<T, ReadOnly> {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_shape_method(&self, _scope: &mut Scope) -> (SequenceExpand<P>, SequenceExpand<O>) {
        unimplemented!("Can't read from tensor map");
    }

    fn __expand_is_in_bounds_method(
        &self,
        _scope: &mut Scope,
        _pos: (SequenceExpand<P>, SequenceExpand<O>),
    ) -> ExpandElementTyped<bool> {
        // Bounds checks are done in hardware, so treat them as always in bounds for the kernels
        true.into()
    }

    #[allow(unused_parens)]
    fn __expand_tensor_map_load_method(
        &self,
        scope: &mut Scope,
        barrier: BarrierExpand,
        shared_memory: SliceExpand<T, ReadWrite>,
        pos: (SequenceExpand<P>, SequenceExpand<O>),
    ) {
        let shared = shared_memory.__expand_try_cast_unchecked_method(scope);
        let (pos, offs) = &pos;
        let rank = pos.len();

        match rank {
            3 => {
                let n = as_i32(scope, pos, 0);
                let w = as_i32(scope, pos, 1);
                let c = as_i32(scope, pos, 2);
                let x = as_u16(scope, offs, 0);
                barrier.__expand_tma_load_im2col_3d_method(scope, self.clone(), shared, n, w, c, x);
            }
            4 => {
                let n = as_i32(scope, pos, 0);
                let h = as_i32(scope, pos, 1);
                let w = as_i32(scope, pos, 2);
                let c = as_i32(scope, pos, 3);
                let y = as_u16(scope, offs, 0);
                let x = as_u16(scope, offs, 1);
                barrier.__expand_tma_load_im2col_4d_method(
                    scope,
                    self.clone(),
                    shared,
                    n,
                    h,
                    w,
                    c,
                    y,
                    x,
                );
            }
            5 => {
                let n = as_i32(scope, pos, 0);
                let d = as_i32(scope, pos, 1);
                let h = as_i32(scope, pos, 2);
                let w = as_i32(scope, pos, 3);
                let c = as_i32(scope, pos, 4);
                let z = as_u16(scope, offs, 0);
                let y = as_u16(scope, offs, 1);
                let x = as_u16(scope, offs, 2);
                barrier.__expand_tma_load_im2col_5d_method(
                    scope,
                    self.clone(),
                    shared,
                    n,
                    d,
                    h,
                    w,
                    c,
                    z,
                    y,
                    x,
                );
            }
            _ => panic!("TMA im2col only supports 3D-5D loads"),
        }
    }
}

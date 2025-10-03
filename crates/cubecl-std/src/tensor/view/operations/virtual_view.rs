use std::marker::PhantomData;

use super::*;
use crate::tensor::layout::{Coordinates, VirtualLayout, VirtualLayoutExpand};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierExpand};

#[derive(CubeType)]
pub struct VirtualView<T: CubePrimitive, C: Coordinates, S: Coordinates, V: ViewOperations<T, S>> {
    #[allow(unused)]
    view: V,
    #[allow(unused)]
    layout: VirtualLayout<C, S>,
    #[cube(comptime)]
    _ty: PhantomData<T>,
}

#[cube]
impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V: ViewOperations<T, S>>
    VirtualView<T, C, S, V>
{
    pub fn new(view: V, layout: VirtualLayout<C, S>) -> Self {
        VirtualView::<T, C, S, V> {
            view,
            layout,
            _ty: PhantomData,
        }
    }
}

impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V: ViewOperations<T, S>>
    VirtualViewExpand<T, C, S, V>
{
    pub fn new(view: V::ExpandType, layout: VirtualLayoutExpand<C, S>) -> Self {
        VirtualViewExpand::<T, C, S, V> {
            view,
            layout,
            _ty: PhantomData,
        }
    }
}

#[derive(CubeType)]
pub struct VirtualViewMut<
    T: CubePrimitive,
    C: Coordinates,
    S: Coordinates,
    V: ViewOperationsMut<T, S>,
> {
    #[allow(unused)]
    view: V,
    #[allow(unused)]
    layout: VirtualLayout<C, S>,
    #[cube(comptime)]
    _ty: PhantomData<T>,
}

#[cube]
impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V: ViewOperationsMut<T, S>>
    VirtualViewMut<T, C, S, V>
{
    pub fn new(view: V, layout: VirtualLayout<C, S>) -> Self {
        VirtualViewMut::<T, C, S, V> {
            view,
            layout,
            _ty: PhantomData,
        }
    }
}

impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V: ViewOperationsMut<T, S>>
    VirtualViewMutExpand<T, C, S, V>
{
    pub fn new(view: V::ExpandType, layout: VirtualLayoutExpand<C, S>) -> Self {
        VirtualViewMutExpand::<T, C, S, V> {
            view,
            layout,
            _ty: PhantomData,
        }
    }
}

macro_rules! impl_virtual_read {
    ($ty: ident, $expand: ident, $trait: ident) => {
        impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> Lined for $ty<T, C, S, V> where
            V: $trait<T, S>
        {
        }
        impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> LinedExpand
            for $expand<T, C, S, V>
        where
            V: $trait<T, S>,
        {
            fn line_size(&self) -> u32 {
                self.view.line_size()
            }
        }

        impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> ViewOperations<T, C>
            for $ty<T, C, S, V>
        where
            V: $trait<T, S>,
        {
        }

        impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> ViewOperationsExpand<T, C>
            for $expand<T, C, S, V>
        where
            V: $trait<T, S>,
        {
            fn __expand_read_method(
                &self,
                scope: &mut Scope,
                pos: <C>::ExpandType,
            ) -> <T>::ExpandType {
                let pos = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.view.clone().__expand_read_method(scope, pos)
            }

            fn __expand_read_checked_method(
                &self,
                scope: &mut Scope,
                pos: <C>::ExpandType,
            ) -> <T>::ExpandType {
                let (read_pos, in_bounds) = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_checked_method(scope, pos);
                let zero = T::__expand_cast_from(scope, 0.into());
                let value = self.view.__expand_read_checked_method(scope, read_pos);
                select::expand::<T>(scope, in_bounds, value, zero)
            }

            fn __expand_read_masked_method(
                &self,
                scope: &mut Scope,
                pos: <C>::ExpandType,
                mask_value: <T>::ExpandType,
            ) -> <T>::ExpandType {
                let (read_pos, in_bounds) = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_checked_method(scope, pos);
                let value = self.view.__expand_read_checked_method(scope, read_pos);
                select::expand::<T>(scope, in_bounds, value, mask_value)
            }

            fn __expand_read_unchecked_method(
                &self,
                scope: &mut Scope,
                pos: <C>::ExpandType,
            ) -> <T>::ExpandType {
                let pos = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.view.__expand_read_unchecked_method(scope, pos)
            }

            fn __expand_to_linear_slice_method(
                &self,
                scope: &mut Scope,
                pos: <C>::ExpandType,
                end: <C>::ExpandType,
            ) -> SliceExpand<T, ReadOnly> {
                let pos = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                let end = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_method(scope, end);
                self.view.__expand_to_linear_slice_method(scope, pos, end)
            }

            fn __expand_shape_method(&self, scope: &mut Scope) -> <C>::ExpandType {
                self.layout.clone().__expand_shape_method(scope)
            }

            fn __expand_is_in_bounds_method(
                &self,
                scope: &mut Scope,
                pos: C::ExpandType,
            ) -> ExpandElementTyped<bool> {
                self.layout.clone().__expand_is_in_bounds_method(scope, pos)
            }

            fn __expand_tensor_map_load_method(
                &self,
                scope: &mut Scope,
                barrier: BarrierExpand,
                shared_memory: SliceExpand<T, ReadWrite>,
                pos: C::ExpandType,
            ) {
                let pos = self
                    .layout
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.view
                    .__expand_tensor_map_load_method(scope, barrier, shared_memory, pos);
            }
        }
    };
}

impl_virtual_read!(VirtualView, VirtualViewExpand, ViewOperations);
impl_virtual_read!(VirtualViewMut, VirtualViewMutExpand, ViewOperationsMut);

impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> ViewOperationsMut<T, C>
    for VirtualViewMut<T, C, S, V>
where
    V: ViewOperationsMut<T, S>,
{
}

impl<T: CubePrimitive, C: Coordinates, S: Coordinates, V> ViewOperationsMutExpand<T, C>
    for VirtualViewMutExpand<T, C, S, V>
where
    V: ViewOperationsMut<T, S>,
{
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        let pos = self
            .layout
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        self.view.__expand_write_method(scope, pos, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        value: <T>::ExpandType,
    ) {
        let (pos, in_bounds) = self
            .layout
            .clone()
            .__expand_to_source_pos_checked_method(scope, pos);
        if_expand(scope, in_bounds.into(), |scope| {
            self.view.__expand_write_checked_method(scope, pos, value);
        });
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        end: <C>::ExpandType,
    ) -> SliceExpand<T, ReadWrite> {
        let pos = self
            .layout
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        let end = self
            .layout
            .clone()
            .__expand_to_source_pos_method(scope, end);
        self.view
            .__expand_to_linear_slice_mut_method(scope, pos, end)
    }

    fn __expand_tensor_map_store_method(
        &self,
        scope: &mut Scope,
        shared_memory: SliceExpand<T, ReadOnly>,
        pos: C::ExpandType,
    ) {
        let pos = self
            .layout
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        self.view
            .__expand_tensor_map_store_method(scope, shared_memory, pos);
    }
}

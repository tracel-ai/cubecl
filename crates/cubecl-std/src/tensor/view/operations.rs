use std::marker::PhantomData;

use cubecl::{prelude::*, unexpanded};
use cubecl_core::{self as cubecl, io::read_masked};

use crate::tensor::layout::{Coordinates, Coords1d, VirtualLayout, VirtualLayoutExpand};

/// Type from which we can read values in cube functions.
/// For a mutable version, see [ListMut].
#[allow(clippy::len_without_is_empty)]
#[cube(self_type = "ref", expand_base_traits = "LinedExpand")]
pub trait ViewOperations<T: CubePrimitive, C: Coordinates>: Lined {
    #[allow(unused)]
    fn read(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_checked(&self, pos: C) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_masked(&self, pos: C, value: T) -> T {
        unexpanded!()
    }

    #[allow(unused)]
    fn read_unchecked(&self, pos: C) -> T {
        unexpanded!()
    }

    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused)]
    fn to_linear_slice(&self, pos: C, size: C) -> Slice<T, ReadOnly> {
        unexpanded!()
    }

    #[allow(unused)]
    fn shape(&self) -> C {
        unexpanded!();
    }

    #[allow(unused)]
    fn is_in_bounds(&self, pos: C) -> bool {
        unexpanded!();
    }
}

/// Type for which we can read and write values in cube functions.
/// For an immutable version, see [List].
#[cube(expand_base_traits = "ViewOperationsExpand<T, C>", self_type = "ref")]
pub trait ViewOperationsMut<T: CubePrimitive, C: Coordinates>: ViewOperations<T, C> {
    #[allow(unused)]
    fn write(&self, pos: C, value: T) {
        unexpanded!()
    }

    #[allow(unused)]
    fn write_checked(&self, pos: C, value: T) {
        unexpanded!()
    }

    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    #[allow(unused, clippy::wrong_self_convention)]
    fn to_linear_slice_mut(&self, pos: C, size: C) -> Slice<T, ReadWrite> {
        unexpanded!()
    }
}

// Automatic implementation for references to List.
impl<'a, T: CubePrimitive, C: Coordinates, V: ViewOperations<T, C>> ViewOperations<T, C> for &'a V
where
    &'a V: CubeType<ExpandType = V::ExpandType>,
{
    fn read(&self, pos: C) -> T {
        V::read(self, pos)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        V::__expand_read(scope, this, pos)
    }
}

// Automatic implementation for mutable references to List.
impl<'a, T: CubePrimitive, C: Coordinates, L: ViewOperations<T, C>> ViewOperations<T, C>
    for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
{
    fn read(&self, pos: C) -> T {
        L::read(self, pos)
    }

    fn __expand_read(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        L::__expand_read(scope, this, pos)
    }
}

// Automatic implementation for references to ListMut.
impl<'a, T: CubePrimitive, C: Coordinates, L: ViewOperationsMut<T, C>> ViewOperationsMut<T, C>
    for &'a L
where
    &'a L: CubeType<ExpandType = L::ExpandType>,
{
    fn write(&self, pos: C, value: T) {
        L::write(self, pos, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, pos, value);
    }
}

// Automatic implementation for mutable references to ListMut.
impl<'a, T: CubePrimitive, C: Coordinates, L: ViewOperationsMut<T, C>> ViewOperationsMut<T, C>
    for &'a mut L
where
    &'a mut L: CubeType<ExpandType = L::ExpandType>,
{
    fn write(&self, pos: C, value: T) {
        L::write(self, pos, value);
    }

    fn __expand_write(
        scope: &mut Scope,
        this: Self::ExpandType,
        pos: C::ExpandType,
        value: T::ExpandType,
    ) {
        L::__expand_write(scope, this, pos, value);
    }
}

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
        }
    };
}

impl_operations_1d!(Array<T>, ExpandElementTyped<Array<T>>);
impl_operations_1d!(Tensor<T>, ExpandElementTyped<Tensor<T>>);
impl_operations_1d!(SharedMemory<T>, ExpandElementTyped<SharedMemory<T>>);

mod slice {
    use super::*;

    impl<T: CubePrimitive, IO: SliceVisibility> ViewOperations<T, Coords1d> for Slice<T, IO> {}
    impl<T: CubePrimitive, IO: SliceVisibility> ViewOperationsExpand<T, Coords1d>
        for SliceExpand<T, IO>
    {
        fn __expand_read_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
        ) -> <T>::ExpandType {
            <Self as ListExpand<T>>::__expand_read_method(self, scope, pos)
        }

        fn __expand_read_checked_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
        ) -> <T>::ExpandType {
            let len = self.__expand_len_method(scope);
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
            let len = self.__expand_len_method(scope);
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
            <Self as ListExpand<T>>::__expand_len_method(self, scope)
        }

        fn __expand_is_in_bounds_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<bool> {
            let len = self.__expand_shape_method(scope);
            lt::expand(scope, pos, len)
        }
    }

    impl<T: CubePrimitive> ViewOperationsMut<T, Coords1d> for Slice<T, ReadWrite> {}
    impl<T: CubePrimitive> ViewOperationsMutExpand<T, Coords1d> for SliceExpand<T, ReadWrite> {
        fn __expand_write_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
            value: <T>::ExpandType,
        ) {
            <Self as ListMutExpand<T>>::__expand_write_method(self, scope, pos, value)
        }

        fn __expand_write_checked_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
            value: <T>::ExpandType,
        ) {
            let len = <Self as ListExpand<T>>::__expand_len_method(self, scope);
            let in_bounds = lt::expand(scope, pos.clone(), len);
            if_expand(scope, in_bounds.into(), |scope| {
                <Self as ListMutExpand<T>>::__expand_write_method(self, scope, pos, value)
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
            <Self as SliceMutOperatorExpand<T>>::__expand_slice_mut_method(self, scope, start, end)
        }
    }
}

mod virtual_tensor {
    use crate::tensor::r#virtual::{VirtualTensor, VirtualTensorExpand};

    use super::*;

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
            let len = self.clone().__expand_buffer_len_method(scope);
            let in_bounds = lt::expand(scope, pos.clone(), len);
            let slice = self.clone().__expand_to_slice_method(scope);
            let zero = Line::__expand_cast_from(scope, 0.into());
            read_masked::expand::<Line<T>>(scope, in_bounds, slice, pos, zero)
        }

        fn __expand_read_masked_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
            mask_value: <Line<T> as CubeType>::ExpandType,
        ) -> <Line<T> as CubeType>::ExpandType {
            let len = self.__expand_len_method(scope);
            let in_bounds = lt::expand(scope, pos.clone(), len);
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
    }
}

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
}

mod view {
    use crate::tensor::{View, ViewExpand};

    use super::*;

    impl<T: CubePrimitive, C: Coordinates, IO: Clone> Lined for View<T, C, IO> {}
    impl<T: CubePrimitive, C: Coordinates, IO: Clone> LinedExpand for ViewExpand<T, C, IO> {
        fn line_size(&self) -> u32 {
            ViewExpand::line_size(self)
        }
    }

    impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperations<T, C> for View<T, C, IO> {}
    impl<T: CubePrimitive, C: Coordinates, IO: Clone> ViewOperationsExpand<T, C>
        for ViewExpand<T, C, IO>
    {
        fn __expand_read_method(&self, scope: &mut Scope, pos: <C>::ExpandType) -> <T>::ExpandType {
            ViewExpand::__expand_read_method(self.clone(), scope, pos)
        }

        fn __expand_read_checked_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
        ) -> <T>::ExpandType {
            ViewExpand::__expand_read_checked_method(self.clone(), scope, pos)
        }

        fn __expand_read_masked_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
            mask_value: <T>::ExpandType,
        ) -> <T>::ExpandType {
            ViewExpand::__expand_read_masked_method(self.clone(), scope, pos, mask_value)
        }

        fn __expand_read_unchecked_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
        ) -> <T>::ExpandType {
            ViewExpand::__expand_read_unchecked_method(self.clone(), scope, pos)
        }

        fn __expand_to_linear_slice_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
            end: <C>::ExpandType,
        ) -> SliceExpand<T, ReadOnly> {
            ViewExpand::__expand_to_linear_slice_inner_method(self.clone(), scope, pos, end)
        }

        fn __expand_shape_method(&self, scope: &mut Scope) -> <C>::ExpandType {
            ViewExpand::__expand_shape_method(self, scope)
        }

        fn __expand_is_in_bounds_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
        ) -> ExpandElementTyped<bool> {
            ViewExpand::__expand_is_in_bounds_method(self, scope, pos)
        }
    }

    impl<T: CubePrimitive, C: Coordinates> ViewOperationsMut<T, C> for View<T, C, ReadWrite> {}
    impl<T: CubePrimitive, C: Coordinates> ViewOperationsMutExpand<T, C>
        for ViewExpand<T, C, ReadWrite>
    {
        fn __expand_write_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
            value: <T>::ExpandType,
        ) {
            ViewExpand::__expand_write_method(self.clone(), scope, pos, value);
        }

        fn __expand_write_checked_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
            value: <T>::ExpandType,
        ) {
            ViewExpand::__expand_write_checked_method(self.clone(), scope, pos, value);
        }

        fn __expand_to_linear_slice_mut_method(
            &self,
            scope: &mut Scope,
            pos: <C>::ExpandType,
            end: <C>::ExpandType,
        ) -> SliceExpand<T, ReadWrite> {
            ViewExpand::__expand_to_linear_slice_mut_inner_method(self.clone(), scope, pos, end)
        }
    }
}

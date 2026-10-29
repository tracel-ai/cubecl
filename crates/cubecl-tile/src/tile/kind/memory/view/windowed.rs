//! A buffer read and written through its [`Window`], the one layer that checks bounds.

use cubecl::prelude::barrier::Barrier;
use cubecl::{prelude::*, unexpanded};
use cubecl_std::tensor::{
    View, ViewExpand, ViewMut, ViewMutExpand, ViewOperations, ViewOperationsExpand,
    ViewOperationsMut, ViewOperationsMutExpand,
    layout::{CoordsDyn, LayoutExpand},
};

use super::super::Window;

/// A buffer addressed through `window`, which alone checks bounds.
///
/// A checked read compares every axis against the buffer once ([`Window::holds`]), reads the
/// buffer unchecked at the position clamped onto its edge, and zeroes what lies past it. Every
/// axis, not only those the window names a boundary on: a window opened at a box larger than its
/// buffer leaves it along an axis that names none. Read as a window layered over the buffer, each
/// layer below would check again, the buffer's every axis and its length, and select the value
/// once per layer.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct WindowedView<'a, T: CubePrimitive> {
    buffer: View<'a, T, CoordsDyn>,
    window: Window,
}

/// The mutable twin of [`WindowedView`]: a checked write lands only inside the window's bounds.
#[expect(dead_code, reason = "written through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct WindowedViewMut<'a, T: CubePrimitive> {
    buffer: ViewMut<'a, T, CoordsDyn>,
    window: Window,
}

#[cube]
impl<'a, T: CubePrimitive> WindowedView<'a, T> {
    pub(crate) fn new(buffer: View<'a, T, CoordsDyn>, window: Window) -> Self {
        WindowedView::<'a, T> { buffer, window }
    }
}

#[cube]
impl<'a, T: CubePrimitive> WindowedViewMut<'a, T> {
    pub(crate) fn new(buffer: ViewMut<'a, T, CoordsDyn>, window: Window) -> Self {
        WindowedViewMut::<'a, T> { buffer, window }
    }
}

impl<'a, T: CubePrimitive> WindowedView<'a, T> {
    /// This view as a plain [`View`].
    pub(crate) fn view(self) -> View<'a, T, CoordsDyn> {
        unexpanded!()
    }

    pub(crate) fn __expand_view(
        scope: &Scope,
        this: WindowedViewExpand<'a, T>,
    ) -> ViewExpand<'a, T, CoordsDyn> {
        this.__expand_view_method(scope)
    }
}

impl<'a, T: CubePrimitive> WindowedViewExpand<'a, T> {
    pub(crate) fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, T, CoordsDyn> {
        ViewExpand::new(scope, self)
    }
}

impl<'a, T: CubePrimitive> WindowedViewMut<'a, T> {
    /// This view as a plain [`ViewMut`].
    pub(crate) fn view_mut(self) -> ViewMut<'a, T, CoordsDyn> {
        unexpanded!()
    }

    pub(crate) fn __expand_view_mut(
        scope: &Scope,
        this: WindowedViewMutExpand<'a, T>,
    ) -> ViewMutExpand<'a, T, CoordsDyn> {
        this.__expand_view_mut_method(scope)
    }
}

impl<'a, T: CubePrimitive> WindowedViewMutExpand<'a, T> {
    pub(crate) fn __expand_view_mut_method(self, scope: &Scope) -> ViewMutExpand<'a, T, CoordsDyn> {
        ViewMutExpand::new(scope, self)
    }
}

/// The position `pos` of `window` lands on in its buffer, and the position a checked read
/// addresses instead: clamped onto the edge on every axis, so it never leaves the buffer,
/// whatever the read then keeps.
fn addressed(
    scope: &Scope,
    window: &<Window as CubeType>::ExpandType,
    pos: <CoordsDyn as CubeType>::ExpandType,
) -> (<CoordsDyn as CubeType>::ExpandType, NativeExpand<bool>) {
    let in_bounds = window
        .clone()
        .__expand_is_in_bounds_method(scope, pos.clone());
    let clamped = window.clone().__expand_edge_clamped_method(scope);
    (clamped.__expand_to_source_pos_method(scope, pos), in_bounds)
}

macro_rules! windowed_reads {
    ($expand: ident, $buffer: ident) => {
        impl<'a, T: CubePrimitive> ViewOperationsExpand<T, CoordsDyn> for $expand<'a, T> {
            fn __expand_read_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
            ) -> T::ExpandType {
                let pos = self
                    .window
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.$buffer.__expand_read_method(scope, pos)
            }

            fn __expand_read_checked_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
            ) -> T::ExpandType {
                let zero = T::__expand_cast_from(scope, 0.into());
                self.__expand_read_masked_method(scope, pos, zero)
            }

            fn __expand_read_masked_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
                mask_value: T::ExpandType,
            ) -> T::ExpandType {
                let (pos, in_bounds) = addressed(scope, &self.window, pos);
                let value = self.$buffer.__expand_read_unchecked_method(scope, pos);
                select::expand::<T>(scope, in_bounds, value, mask_value)
            }

            fn __expand_read_unchecked_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
            ) -> T::ExpandType {
                let pos = self
                    .window
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.$buffer.__expand_read_unchecked_method(scope, pos)
            }

            fn __expand_as_linear_slice_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
                end: <CoordsDyn as CubeType>::ExpandType,
            ) -> &SliceExpand<T> {
                let pos = self
                    .window
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                let end = self
                    .window
                    .clone()
                    .__expand_to_source_pos_method(scope, end);
                ViewOperationsExpand::<T, CoordsDyn>::__expand_as_linear_slice_method(
                    &self.$buffer,
                    scope,
                    pos,
                    end,
                )
            }

            fn __expand_tensor_map_load_method(
                &self,
                scope: &Scope,
                barrier: &NativeExpand<Barrier>,
                shared_memory: &mut SliceExpand<T>,
                pos: <CoordsDyn as CubeType>::ExpandType,
            ) {
                let pos = self
                    .window
                    .clone()
                    .__expand_to_source_pos_method(scope, pos);
                self.$buffer
                    .__expand_tensor_map_load_method(scope, barrier, shared_memory, pos)
            }

            fn __expand_shape_method(&self, scope: &Scope) -> <CoordsDyn as CubeType>::ExpandType {
                self.window.clone().__expand_shape_method(scope)
            }

            fn __expand_is_in_bounds_method(
                &self,
                scope: &Scope,
                pos: <CoordsDyn as CubeType>::ExpandType,
            ) -> NativeExpand<bool> {
                self.window.clone().__expand_is_in_bounds_method(scope, pos)
            }
        }

        impl<'a, T: CubePrimitive> VectorizedExpand for $expand<'a, T> {
            fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
                self.$buffer.__expand_vector_size_method(scope)
            }
        }
    };
}

windowed_reads!(WindowedViewExpand, buffer);
windowed_reads!(WindowedViewMutExpand, buffer);

impl<'a, T: CubePrimitive> Vectorized for WindowedView<'a, T> {}
impl<'a, T: CubePrimitive> Vectorized for WindowedViewMut<'a, T> {}
impl<'a, T: CubePrimitive> ViewOperations<T, CoordsDyn> for WindowedView<'a, T> {}
impl<'a, T: CubePrimitive> ViewOperations<T, CoordsDyn> for WindowedViewMut<'a, T> {}
impl<'a, T: CubePrimitive> ViewOperationsMut<T, CoordsDyn> for WindowedViewMut<'a, T> {}

impl<'a, T: CubePrimitive> ViewOperationsMutExpand<T, CoordsDyn> for WindowedViewMutExpand<'a, T> {
    fn __expand_write_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
        value: T::ExpandType,
    ) {
        let pos = self
            .window
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        self.buffer.__expand_write_method(scope, pos, value);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
        value: T::ExpandType,
    ) {
        let in_bounds = self
            .window
            .clone()
            .__expand_is_in_bounds_method(scope, pos.clone());
        let pos = self
            .window
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        if_expand(scope, in_bounds, |scope| {
            self.buffer.__expand_write_method(scope, pos, value);
        });
    }

    fn __expand_as_linear_slice_mut_method(
        &self,
        scope: &Scope,
        pos: <CoordsDyn as CubeType>::ExpandType,
        end: <CoordsDyn as CubeType>::ExpandType,
    ) -> &mut SliceExpand<T> {
        let pos = self
            .window
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        let end = self
            .window
            .clone()
            .__expand_to_source_pos_method(scope, end);
        ViewOperationsMutExpand::<T, CoordsDyn>::__expand_as_linear_slice_mut_method(
            &self.buffer,
            scope,
            pos,
            end,
        )
    }

    fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &SliceExpand<T>,
        pos: <CoordsDyn as CubeType>::ExpandType,
    ) -> PendingExpand<()> {
        let pos = self
            .window
            .clone()
            .__expand_to_source_pos_method(scope, pos);
        self.buffer
            .__expand_tensor_map_store_method(scope, shared_memory, pos)
    }
}

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::VectorSize, prelude::barrier::Barrier, unexpanded};

use crate::tensor::{
    ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand, VirtualView,
    VirtualViewMut,
    layout::{Coordinates, Layout, VirtualLayout, VirtualLayoutExpand, slice::SliceLayout},
};

/// A conceptual view of an underlying linear storage.
/// Allows abstract indexing in multiple dimensions, without having to know the data layout or
/// location.
#[derive(Clone, Copy)]
pub struct View<'a, E: CubePrimitive, C: Coordinates> {
    _layout: PhantomData<C>,
    _ty: PhantomData<E>,
    _lifetime: PhantomData<&'a ()>,
}

/// Expand type of [`View`]
#[derive(Clone, Copy)]
pub struct ViewExpand<'a, E: CubePrimitive, C: Coordinates> {
    pub(super) inner: &'a (dyn ViewOperationsExpand<E, C> + 'a),
}

/// Mutable view
/// Note: `Clone` and `Copy` should ideally not be there, but are required for good ergonomics until
/// `Reborrow` and `CoerceShared` are implemented and stabilized.
#[derive(Clone, Copy)]
pub struct ViewMut<'a, E: CubePrimitive, C: Coordinates> {
    _layout: PhantomData<C>,
    _ty: PhantomData<E>,
    _lifetime: PhantomData<&'a mut ()>,
}

/// Expand type of [`ViewMutExpand`]
#[derive(Clone, Copy)]
pub struct ViewMutExpand<'a, E: CubePrimitive, C: Coordinates> {
    pub(super) inner: &'a (dyn ViewOperationsMutExpand<E, C> + 'a),
}

macro_rules! impl_cube_type {
    ($ty: ident, $expand: ident) => {
        impl<'a, E: CubePrimitive, C: Coordinates + 'a> CubeType for $ty<'a, E, C> {
            type ExpandType = $expand<'a, E, C>;
        }

        impl<'a, E: CubePrimitive, C: Coordinates> IntoExpand for $expand<'a, E, C> {
            type Expand = $expand<'a, E, C>;

            fn into_expand(self, _: &Scope) -> Self::Expand {
                self
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> ExpandTypeClone for $expand<'a, E, C> {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> IntoMut for $expand<'a, E, C> {
            fn into_mut(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> CubeDebug for $expand<'a, E, C> {}

        impl<'a, E: CubePrimitive, C: Coordinates> AsRefExpand for $expand<'a, E, C> {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl<'a, E: CubePrimitive, C: Coordinates> AsMutExpand for $expand<'a, E, C> {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
        impl<'a, E: CubePrimitive, C: Coordinates> DerefExpand for $expand<'a, E, C> {
            type Target = Self;

            fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
                self.clone()
            }
        }
    };
}

impl_cube_type!(View, ViewExpand);
impl_cube_type!(ViewMut, ViewMutExpand);

impl<'a, E: CubePrimitive, C: Coordinates + 'a> View<'a, E, C> {
    /// Create a new tensor view from an underlying concrete storage and a layout to map it into
    /// the target coordinate space
    #[allow(unused_variables)]
    pub fn new<V: ViewOperations<E, S> + 'a, S: Coordinates>(
        view: V,
        layout: impl Into<VirtualLayout<C, S>>,
    ) -> Self {
        View {
            _layout: PhantomData,
            _ty: PhantomData,
            _lifetime: PhantomData,
        }
    }

    /// Expand function for [`View::new`]
    pub fn __expand_new<V: ViewOperations<E, S> + 'a, S: Coordinates + 'a>(
        scope: &Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewExpand<'a, E, C> {
        ViewExpand::new(
            scope,
            VirtualView::<E, C, S, V>::__expand_new(scope, view, layout),
        )
    }
}

impl<'a, E: CubePrimitive, C: Coordinates + 'a> ViewMut<'a, E, C> {
    /// Create a new tensor view from an underlying concrete storage and a layout to map it into
    /// the target coordinate space
    #[allow(unused_variables)]
    pub fn new<V: ViewOperationsMut<E, S> + 'a, S: Coordinates>(
        view: V,
        layout: impl Into<VirtualLayout<C, S>>,
    ) -> Self {
        ViewMut {
            _layout: PhantomData,
            _ty: PhantomData,
            _lifetime: PhantomData,
        }
    }

    /// Expand function for [`View::new`]
    pub fn __expand_new<V: ViewOperationsMut<E, S> + 'a, S: Coordinates + 'a>(
        scope: &Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewMutExpand<'a, E, C> {
        ViewMutExpand::new(
            scope,
            VirtualViewMut::<E, C, S, V>::__expand_new(scope, view, layout),
        )
    }
}

macro_rules! impl_read {
    ($ty: ident, $expand: ident) => {
        impl<'a, E: CubePrimitive, C: Coordinates + 'a> $ty<'a, E, C> {
            pub fn view<T: Coordinates + 'a>(
                self,
                _layout: impl Into<VirtualLayout<T, C>>,
            ) -> $ty<'a, E, T> {
                unexpanded!()
            }

            pub fn __expand_view<T: Coordinates + 'a>(
                scope: &Scope,
                this: $expand<'a, E, C>,
                layout: VirtualLayoutExpand<T, C>,
            ) -> $expand<'a, E, T> {
                this.__expand_view_method(scope, layout)
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates + 'a> $expand<'a, E, C> {
            pub fn __expand_view_method<T: Coordinates + 'a>(
                self,
                scope: &Scope,
                layout: VirtualLayoutExpand<T, C>,
            ) -> $expand<'a, E, T> {
                $ty::__expand_new::<$ty<'a, E, C>, C>(scope, self, layout)
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> $ty<'a, E, C> {
            /// Calls [`Layout::shape`] on the view's layout
            pub fn shape(&self) -> C {
                unexpanded!()
            }

            /// Calls [`Layout::is_in_bounds`] on the view's layout
            pub fn is_in_bounds(&self, _pos: C) -> bool {
                unexpanded!()
            }

            pub fn __expand_shape(scope: &Scope, this: $expand<'a, E, C>) -> C::ExpandType {
                this.__expand_shape_method(scope)
            }

            pub fn __expand_is_in_bounds(
                scope: &Scope,
                this: $expand<'a, E, C>,
                pos: C::ExpandType,
            ) -> NativeExpand<bool> {
                this.__expand_is_in_bounds_method(scope, pos)
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> $expand<'a, E, C> {
            pub fn __expand_shape_method(&self, scope: &Scope) -> C::ExpandType {
                self.inner.__expand_shape_method(scope)
            }

            pub fn __expand_is_in_bounds_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> NativeExpand<bool> {
                self.inner.__expand_is_in_bounds_method(scope, pos)
            }
        }

        #[allow(unused_variables)]
        impl<'a, E: CubePrimitive, C: Coordinates> $ty<'a, E, C> {
            /// Read a value at `pos`. The layout handles translation into a concrete index.
            pub fn read(&self, pos: C) -> E {
                unexpanded!()
            }

            /// Read a value at `pos`. The layout handles translation into a concrete index.
            /// Reading is done unchecked
            pub fn read_unchecked(&self, pos: C) -> E {
                unexpanded!()
            }

            /// Read a value at `pos` if it's in bounds. The layout handles translation into a concrete index.
            pub fn read_checked(&self, pos: C) -> E {
                unexpanded!()
            }

            /// Read a value at `pos` if it's in bounds, returning `mask_value` otherwise. The layout handles translation into a concrete index.
            pub fn read_masked(&self, pos: C, mask_value: E) -> E {
                unexpanded!()
            }

            /// Interpret this view as a linear slice encompassing the entire view.
            ///
            /// # Safety
            ///
            /// No checking is done on whether the slice is contiguous in memory.
            pub fn as_linear_slice(&self) -> &'a [E] {
                unexpanded!()
            }

            pub fn vector_size(&self) -> VectorSize {
                unexpanded!()
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> $expand<'a, E, C> {
            /// Expand method for [`View::read`]
            pub fn __expand_read_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> NativeExpand<E> {
                self.inner.__expand_read_method(scope, pos)
            }

            /// Expand method for [`View::read_unchecked`]
            pub fn __expand_read_unchecked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> NativeExpand<E> {
                self.inner.__expand_read_unchecked_method(scope, pos)
            }

            /// Expand method for [`View::read_checked`]
            pub fn __expand_read_checked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
            ) -> NativeExpand<E> {
                self.inner.__expand_read_checked_method(scope, pos)
            }

            /// Expand method for [`View::read_masked`]
            pub fn __expand_read_masked_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
                mask_value: E::ExpandType,
            ) -> NativeExpand<E> {
                self.inner
                    .__expand_read_masked_method(scope, pos, mask_value)
            }

            /// Expand method for [`View::vector_size`]
            pub fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
                self.inner.vector_size()
            }

            pub fn vector_size(&self) -> VectorSize {
                self.inner.vector_size()
            }

            pub fn __expand_as_linear_slice_method(&self, scope: &Scope) -> &'a SliceExpand<E> {
                let shape = self.inner.__expand_shape_method(scope);
                let origin = C::__expand_from_int(scope, shape.clone_unchecked(), 0);
                // Inclusive end so clamping works correctly
                let one = C::__expand_from_int(scope, shape.clone_unchecked(), 1);
                let shape = C::__expand_max(scope, shape, one.clone_unchecked());
                let end = C::__expand_sub(scope, shape, one);
                let slice = self
                    .inner
                    .__expand_as_linear_slice_method(scope, origin, end);
                scope.create_kernel_ref(slice.expand.into())
            }

            pub(super) fn __expand_as_linear_slice_inner_method(
                &self,
                scope: &Scope,
                pos: C::ExpandType,
                end: C::ExpandType,
            ) -> &SliceExpand<E> {
                self.inner.__expand_as_linear_slice_method(scope, pos, end)
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates + 'static> $ty<'a, E, C> {
            /// Create a slice starting from `pos`, with `size`.
            /// The layout handles translation into concrete indices.
            /// Size will be clamped to the current layout size.
            pub fn slice(self, _pos: C, _size: C) -> $ty<'a, E, C> {
                unexpanded!()
            }

            /// Create a slice starting from `pos`, with `size`.
            /// The layout handles translation into concrete indices.
            /// Size and pos will be clamped to the current layout size.
            /// #Safety
            /// Access is always unchecked
            pub fn slice_unchecked(self, _pos: C, _size: C) -> $ty<'a, E, C> {
                unexpanded!()
            }

            pub fn __expand_slice(
                scope: &Scope,
                this: $expand<'a, E, C>,
                pos: C::ExpandType,
                size: C::ExpandType,
            ) -> $expand<'a, E, C> {
                this.__expand_slice_method(scope, pos, size)
            }

            pub fn __expand_slice_unchecked(
                scope: &Scope,
                this: $expand<'a, E, C>,
                pos: C::ExpandType,
                size: C::ExpandType,
            ) -> $expand<'a, E, C> {
                this.__expand_slice_unchecked_method(scope, pos, size)
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates + 'static> $expand<'a, E, C> {
            pub fn __expand_slice_method(
                self,
                scope: &Scope,
                pos: C::ExpandType,
                size: C::ExpandType,
            ) -> $expand<'a, E, C> {
                self.slice(scope, pos, size, true)
            }

            pub fn __expand_slice_unchecked_method(
                self,
                scope: &Scope,
                pos: C::ExpandType,
                size: C::ExpandType,
            ) -> $expand<'a, E, C> {
                self.slice(scope, pos, size, false)
            }

            fn slice(
                self,
                scope: &Scope,
                pos: C::ExpandType,
                size: C::ExpandType,
                checked: bool,
            ) -> $expand<'a, E, C> {
                let shape = self.__expand_shape_method(scope);
                let pos = C::__expand_min(scope, pos, shape.clone_unchecked());
                let max_size = C::__expand_sub(scope, shape, pos.clone_unchecked());
                let size = C::__expand_min(scope, size, max_size);
                let layout = SliceLayout::__expand_new(scope, pos, size, checked);
                $ty::__expand_new::<$ty<'a, E, C>, _>(scope, self, layout.into())
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates + 'static> $ty<'a, E, C> {
            ///.Execute a TMA load into shared memory, if the underlying storage supports it.
            /// Panics if it's unsupported.
            pub fn tensor_map_load(&self, _barrier: &Barrier, _shared_memory: &mut [E], _pos: C) {
                unexpanded!()
            }
        }

        impl<'a, E: CubePrimitive, C: Coordinates> $expand<'a, E, C> {
            pub fn __expand_tensor_map_load_method(
                &self,
                scope: &Scope,
                barrier: &NativeExpand<Barrier>,
                shared_memory: &mut SliceExpand<E>,
                pos: C::ExpandType,
            ) {
                self.inner
                    .__expand_tensor_map_load_method(scope, barrier, shared_memory, pos)
            }
        }
    };
}

impl_read!(View, ViewExpand);
impl_read!(ViewMut, ViewMutExpand);

impl<'a, E: CubePrimitive, C: Coordinates> ViewExpand<'a, E, C> {
    pub fn new<V: ViewOperationsExpand<E, C> + 'a>(scope: &Scope, view: V) -> Self {
        let inner: &dyn ViewOperationsExpand<E, C> = scope.create_kernel_ref(view);
        ViewExpand { inner }
    }
}

impl<'a, E: CubePrimitive, C: Coordinates> ViewMutExpand<'a, E, C> {
    pub fn new<V: ViewOperationsMutExpand<E, C> + 'a>(scope: &Scope, view: V) -> Self {
        let inner: &mut dyn ViewOperationsMutExpand<E, C> = scope.create_kernel_ref(view);
        ViewMutExpand { inner }
    }
}

impl<'a, E: CubePrimitive, C: Coordinates + 'a> ViewMut<'a, E, C> {
    pub fn view_mut<'b, T: Coordinates + 'a>(
        self,
        _layout: impl Layout<Coordinates = T, SourceCoordinates = C>,
    ) -> ViewMut<'b, E, T>
    where
        'a: 'b,
    {
        unexpanded!()
    }

    pub fn __expand_view_mut<T: Coordinates + 'a>(
        scope: &Scope,
        this: ViewMutExpand<'a, E, C>,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewMutExpand<'a, E, T> {
        this.__expand_view_mut_method(scope, layout)
    }
}

impl<'a, E: CubePrimitive, C: Coordinates + 'a> ViewMutExpand<'a, E, C> {
    pub fn __expand_view_mut_method<'b, T: Coordinates + 'a>(
        self,
        scope: &Scope,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewMutExpand<'b, E, T>
    where
        'a: 'b,
    {
        ViewMut::__expand_new::<ViewMut<'a, E, C>, C>(scope, self, layout)
    }
}

#[allow(unused_variables)]
impl<'a, E: CubePrimitive, C: Coordinates> ViewMut<'a, E, C> {
    /// Write a value to `pos`. The layout handles translation into a concrete index.
    pub fn write(&mut self, pos: C, value: E) {
        unexpanded!()
    }

    /// Write a value to `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn write_checked(&mut self, pos: C, value: E) {
        unexpanded!()
    }

    /// Interpret this view as a mutable linear slice encompassing the entire view.
    ///
    /// # Safety
    ///
    /// No checking is done on whether the slice is contiguous in memory.
    pub fn to_linear_slice_mut(&mut self) -> &'a mut [E] {
        unexpanded!()
    }
}

impl<'a, E: CubePrimitive, C: Coordinates> ViewMutExpand<'a, E, C> {
    /// Expand method for [`View::write`]
    pub fn __expand_write_method(
        &mut self,
        scope: &Scope,
        pos: C::ExpandType,
        value: NativeExpand<E>,
    ) {
        self.inner.__expand_write_method(scope, pos, value)
    }

    /// Expand method for [`View::write_checked`]
    pub fn __expand_write_checked_method(
        &mut self,
        scope: &Scope,
        pos: C::ExpandType,
        value: NativeExpand<E>,
    ) {
        self.inner.__expand_write_checked_method(scope, pos, value);
    }

    pub fn __expand_as_linear_slice_mut_method(&mut self, scope: &Scope) -> &'a mut SliceExpand<E> {
        let shape = self.inner.__expand_shape_method(scope);
        let origin = C::__expand_from_int(scope, shape.clone_unchecked(), 0);
        // Inclusive end so clamping works correctly
        let one = C::__expand_from_int(scope, shape.clone_unchecked(), 1);
        let shape = C::__expand_max(scope, shape, one.clone_unchecked());
        let end = C::__expand_sub(scope, shape, one);
        let slice = self
            .inner
            .__expand_as_linear_slice_mut_method(scope, origin, end);
        scope.create_kernel_ref(slice.expand.into())
    }

    pub(super) fn __expand_to_linear_slice_mut_inner_method(
        &mut self,
        scope: &Scope,
        pos: C::ExpandType,
        end: C::ExpandType,
    ) -> &mut SliceExpand<E> {
        self.inner
            .__expand_as_linear_slice_mut_method(scope, pos, end)
    }
}

impl<'a, E: CubePrimitive, C: Coordinates + 'static> ViewMut<'a, E, C> {
    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size and pos will be clamped to the current layout size.
    pub fn slice_mut(self, _pos: C, _size: C) -> ViewMut<'a, E, C> {
        unexpanded!()
    }

    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size and pos will be clamped to the current layout size.
    ///
    /// # Safety
    /// Access is always unchecked.
    pub fn slice_mut_unchecked(self, _pos: C, _size: C) -> ViewMut<'a, E, C> {
        unexpanded!()
    }

    pub fn __expand_slice_mut(
        scope: &Scope,
        this: ViewMutExpand<'a, E, C>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewMutExpand<'a, E, C> {
        this.__expand_slice_mut_method(scope, pos, size)
    }

    pub fn __expand_slice_mut_unchecked(
        scope: &Scope,
        this: ViewMutExpand<'a, E, C>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewMutExpand<'a, E, C> {
        this.__expand_slice_mut_unchecked_method(scope, pos, size)
    }
}

impl<'a, E: CubePrimitive, C: Coordinates + 'static> ViewMutExpand<'a, E, C> {
    pub fn __expand_slice_mut_method(
        self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewMutExpand<'a, E, C> {
        self.slice_mut(scope, pos, size, true)
    }

    pub fn __expand_slice_mut_unchecked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewMutExpand<'a, E, C> {
        self.slice_mut(scope, pos, size, false)
    }

    fn slice_mut(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
        checked: bool,
    ) -> ViewMutExpand<'a, E, C> {
        let shape = self.__expand_shape_method(scope);
        let pos = C::__expand_min(scope, pos, shape.clone_unchecked());
        let max_size = C::__expand_sub(scope, shape, pos.clone_unchecked());
        let size = C::__expand_min(scope, size, max_size);
        let layout = SliceLayout::__expand_new(scope, pos, size, checked);
        self.clone().__expand_view_mut_method(scope, layout.into())
    }
}

impl<'a, E: CubePrimitive, C: Coordinates> ViewMut<'a, E, C> {
    ///.Execute a TMA store into global memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    pub fn tensor_map_store(&self, _shared_memory: &[E], _pos: C) {
        unexpanded!()
    }
}

impl<'a, E: CubePrimitive, C: Coordinates> ViewMutExpand<'a, E, C> {
    pub fn __expand_tensor_map_store_method(
        &mut self,
        scope: &Scope,
        shared_memory: &SliceExpand<E>,
        pos: C::ExpandType,
    ) {
        self.inner
            .__expand_tensor_map_store_method(scope, shared_memory, pos)
    }
}

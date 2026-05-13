use std::{marker::PhantomData, sync::Arc};

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
pub struct View<E: CubePrimitive, C: Coordinates, IO: Clone = ReadOnly> {
    _layout: PhantomData<C>,
    _ty: PhantomData<(E, IO)>,
}

// `View` is a dummy type so it's always send/sync
unsafe impl<E: CubePrimitive, C: Coordinates, IO: Clone> Send for View<E, C, IO> {}
unsafe impl<E: CubePrimitive, C: Coordinates, IO: Clone> Sync for View<E, C, IO> {}

#[derive(Clone)]
pub(super) enum ViewType<E: CubePrimitive, C: Coordinates> {
    Read(Arc<dyn ViewOperationsExpand<E, C>>),
    ReadWrite(Arc<dyn ViewOperationsMutExpand<E, C>>),
}

impl<E: CubePrimitive, C: Coordinates> ViewType<E, C> {
    /// Dereference in read mode
    pub fn read(&self) -> &dyn ViewOperationsExpand<E, C> {
        match self {
            ViewType::Read(list) => &**list,
            ViewType::ReadWrite(list) => &**list,
        }
    }

    /// Dereference in write mode
    pub fn write(&self) -> &dyn ViewOperationsMutExpand<E, C> {
        match self {
            ViewType::Read(_) => panic!("Can't write to readonly list"),
            ViewType::ReadWrite(list) => &**list,
        }
    }
}

/// Expand type of [`View`]
#[derive(Clone)]
pub struct ViewExpand<E: CubePrimitive, C: Coordinates, IO: Clone = ReadOnly> {
    pub(super) inner: ViewType<E, C>,
    pub(super) _io: PhantomData<IO>,
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeType for View<E, C, IO> {
    type ExpandType = ViewExpand<E, C, IO>;
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> IntoExpand for ViewExpand<E, C, IO> {
    type Expand = ViewExpand<E, C, IO>;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ExpandTypeClone for ViewExpand<E, C, IO> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> IntoMut for ViewExpand<E, C, IO> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeDebug for ViewExpand<E, C, IO> {}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> AsRefExpand for ViewExpand<E, C, IO> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<E: CubePrimitive, C: Coordinates, IO: Clone> AsMutExpand for ViewExpand<E, C, IO> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
impl<E: CubePrimitive, C: Coordinates, IO: Clone> DerefExpand for ViewExpand<E, C, IO> {
    type Target = Self;

    fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
        self.clone()
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadOnly> {
    /// Create a new tensor view from an underlying concrete storage and a layout to map it into
    /// the target coordinate space
    #[allow(unused_variables)]
    pub fn new<V: ViewOperations<E, S>, S: Coordinates>(
        view: V,
        layout: impl Into<VirtualLayout<C, S>>,
    ) -> Self {
        View {
            _layout: PhantomData,
            _ty: PhantomData,
        }
    }

    /// Expand function for [`View::new`]
    pub fn __expand_new<V: ViewOperations<E, S> + 'static, S: Coordinates + 'static>(
        scope: &Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewExpand<E, C, ReadOnly> {
        ViewExpand::new(VirtualView::<E, C, S, V>::__expand_new(scope, view, layout))
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> View<E, C, IO> {
    pub fn view<T: Coordinates>(
        &self,
        _layout: impl Into<VirtualLayout<T, C>>,
    ) -> View<E, T, ReadOnly> {
        unexpanded!()
    }

    pub fn __expand_view<T: Coordinates + 'static>(
        scope: &Scope,
        this: ViewExpand<E, C, IO>,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadOnly> {
        this.__expand_view_method(scope, layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> ViewExpand<E, C, IO> {
    pub fn __expand_view_method<T: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadOnly> {
        View::__expand_new::<View<E, C, IO>, C>(scope, self.clone(), layout)
    }

    pub fn new<V: ViewOperationsExpand<E, C> + 'static>(view: V) -> Self {
        ViewExpand {
            inner: ViewType::Read(Arc::new(view)),
            _io: PhantomData,
        }
    }

    pub fn new_mut<V: ViewOperationsMutExpand<E, C> + 'static>(view: V) -> Self {
        ViewExpand {
            inner: ViewType::ReadWrite(Arc::new(view)),
            _io: PhantomData,
        }
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadWrite> {
    pub fn view_mut<T: Coordinates>(
        &self,
        _layout: impl Layout<Coordinates = T, SourceCoordinates = C>,
    ) -> View<E, T, ReadWrite> {
        unexpanded!()
    }

    pub fn __expand_view_mut<T: Coordinates + 'static>(
        scope: &Scope,
        this: ViewExpand<E, C, ReadWrite>,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadWrite> {
        this.__expand_view_mut_method(scope, layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> ViewExpand<E, C, ReadWrite> {
    pub fn __expand_view_mut_method<T: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadWrite> {
        View::__expand_new_mut::<View<E, C, ReadWrite>, C>(scope, self.clone(), layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadWrite> {
    /// Create a new mutable tensor view from an underlying concrete storage and a layout to map it
    /// into the target coordinate space
    pub fn new_mut<V: ViewOperationsMut<E, S>, S: Coordinates>(
        _view: V,
        _layout: impl Into<VirtualLayout<C, S>>,
    ) -> View<E, C, ReadWrite> {
        View {
            _ty: PhantomData,
            _layout: PhantomData,
        }
    }

    /// Expand function for [`View::new_mut`]
    pub fn __expand_new_mut<V: ViewOperationsMut<E, S> + 'static, S: Coordinates + 'static>(
        scope: &Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewExpand<E, C, ReadWrite> {
        ViewExpand::new_mut(VirtualViewMut::<E, C, S, V>::__expand_new(
            scope, view, layout,
        ))
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> View<E, C, IO> {
    /// Calls [`Layout::shape`] on the view's layout
    pub fn shape(&self) -> C {
        unexpanded!()
    }

    /// Calls [`Layout::is_in_bounds`] on the view's layout
    pub fn is_in_bounds(&self, _pos: C) -> bool {
        unexpanded!()
    }

    pub fn __expand_shape(scope: &Scope, this: ViewExpand<E, C, IO>) -> C::ExpandType {
        this.__expand_shape_method(scope)
    }

    pub fn __expand_is_in_bounds(
        scope: &Scope,
        this: ViewExpand<E, C, IO>,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        this.__expand_is_in_bounds_method(scope, pos)
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ViewExpand<E, C, IO> {
    pub fn __expand_shape_method(&self, scope: &Scope) -> C::ExpandType {
        self.inner.read().__expand_shape_method(scope)
    }

    pub fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        self.inner.read().__expand_is_in_bounds_method(scope, pos)
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates, IO: Clone> View<E, C, IO> {
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
    pub fn as_linear_slice(&self) -> &[E] {
        unexpanded!()
    }

    pub fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ViewExpand<E, C, IO> {
    /// Expand method for [`View::read`]
    pub fn __expand_read_method(&self, scope: &Scope, pos: C::ExpandType) -> NativeExpand<E> {
        self.inner.read().__expand_read_method(scope, pos)
    }

    /// Expand method for [`View::read_unchecked`]
    pub fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<E> {
        self.inner.read().__expand_read_unchecked_method(scope, pos)
    }

    /// Expand method for [`View::read_checked`]
    pub fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<E> {
        self.inner.read().__expand_read_checked_method(scope, pos)
    }

    /// Expand method for [`View::read_masked`]
    pub fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        mask_value: E::ExpandType,
    ) -> NativeExpand<E> {
        self.inner
            .read()
            .__expand_read_masked_method(scope, pos, mask_value)
    }

    /// Expand method for [`View::vector_size`]
    pub fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.inner.read().vector_size()
    }

    pub fn vector_size(&self) -> VectorSize {
        self.inner.read().vector_size()
    }

    pub fn __expand_as_linear_slice_method(&self, scope: &Scope) -> &SliceExpand<E> {
        let shape = self.inner.read().__expand_shape_method(scope);
        let origin = C::__expand_from_int(scope, shape.clone_unchecked(), 0);
        // Inclusive end so clamping works correctly
        let one = C::__expand_from_int(scope, shape.clone_unchecked(), 1);
        let shape = C::__expand_max(scope, shape, one.clone_unchecked());
        let end = C::__expand_sub(scope, shape, one);
        self.inner
            .read()
            .__expand_as_linear_slice_method(scope, origin, end)
    }

    pub(super) fn __expand_as_linear_slice_inner_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        end: C::ExpandType,
    ) -> &SliceExpand<E> {
        self.inner
            .read()
            .__expand_as_linear_slice_method(scope, pos, end)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> View<E, C, IO> {
    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size will be clamped to the current layout size.
    pub fn slice(&self, _pos: C, _size: C) -> &View<E, C, ReadOnly> {
        unexpanded!()
    }

    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size and pos will be clamped to the current layout size.
    /// #Safety
    /// Access is always unchecked
    pub fn slice_unchecked(&self, _pos: C, _size: C) -> &View<E, C, ReadOnly> {
        unexpanded!()
    }

    pub fn __expand_slice<'this>(
        scope: &Scope,
        this: &'this ViewExpand<E, C, IO>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> &'this ViewExpand<E, C, ReadOnly> {
        this.__expand_slice_method(scope, pos, size)
    }

    pub fn __expand_slice_unchecked<'this>(
        scope: &Scope,
        this: &'this ViewExpand<E, C, IO>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> &'this ViewExpand<E, C, ReadOnly> {
        this.__expand_slice_unchecked_method(scope, pos, size)
    }
}

#[cube]
impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> View<E, C, IO> {}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> ViewExpand<E, C, IO> {
    pub fn __expand_slice_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> &ViewExpand<E, C, ReadOnly> {
        self.slice(scope, pos, size, true)
    }

    pub fn __expand_slice_unchecked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> &ViewExpand<E, C, ReadOnly> {
        self.slice(scope, pos, size, false)
    }

    fn slice(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
        checked: bool,
    ) -> &ViewExpand<E, C, ReadOnly> {
        let shape = self.__expand_shape_method(scope);
        let pos = C::__expand_min(scope, pos, shape.clone_unchecked());
        let max_size = C::__expand_sub(scope, shape, pos.clone_unchecked());
        let size = C::__expand_min(scope, size, max_size);
        let layout = SliceLayout::__expand_new(scope, pos, size, checked);
        let view = self.__expand_view_method(scope, layout.into());
        scope.create_kernel_ref(view)
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates> View<E, C, ReadWrite> {
    /// Write a value to `pos`. The layout handles translation into a concrete index.
    pub fn write(&self, pos: C, value: E) {
        unexpanded!()
    }

    /// Write a value to `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn write_checked(&self, pos: C, value: E) {
        unexpanded!()
    }

    /// Interpret this view as a mutable linear slice encompassing the entire view.
    ///
    /// # Safety
    ///
    /// No checking is done on whether the slice is contiguous in memory.
    pub fn to_linear_slice_mut(&self) -> &mut [E] {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates> ViewExpand<E, C, ReadWrite> {
    /// Expand method for [`View::write`]
    pub fn __expand_write_method(&self, scope: &Scope, pos: C::ExpandType, value: NativeExpand<E>) {
        self.inner.write().__expand_write_method(scope, pos, value)
    }

    /// Expand method for [`View::write_checked`]
    pub fn __expand_write_checked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        value: NativeExpand<E>,
    ) {
        self.inner
            .write()
            .__expand_write_checked_method(scope, pos, value);
    }

    pub fn __expand_as_linear_slice_mut_method(&mut self, scope: &Scope) -> &mut SliceExpand<E> {
        let shape = self.inner.read().__expand_shape_method(scope);
        let origin = C::__expand_from_int(scope, shape.clone_unchecked(), 0);
        // Inclusive end so clamping works correctly
        let one = C::__expand_from_int(scope, shape.clone_unchecked(), 1);
        let shape = C::__expand_max(scope, shape, one.clone_unchecked());
        let end = C::__expand_sub(scope, shape, one);
        self.inner
            .write()
            .__expand_as_linear_slice_mut_method(scope, origin, end)
    }

    pub(super) fn __expand_to_linear_slice_mut_inner_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        end: C::ExpandType,
    ) -> &mut SliceExpand<E> {
        self.inner
            .write()
            .__expand_as_linear_slice_mut_method(scope, pos, end)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadWrite> {
    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size and pos will be clamped to the current layout size.
    pub fn slice_mut(&self, _pos: C, _size: C) -> View<E, C, ReadWrite> {
        unexpanded!()
    }

    /// Create a mutable slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    /// Size and pos will be clamped to the current layout size.
    ///
    /// # Safety
    /// Access is always unchecked.
    pub fn slice_mut_unchecked(&self, _pos: C, _size: C) -> View<E, C, ReadWrite> {
        unexpanded!()
    }

    pub fn __expand_slice_mut(
        scope: &Scope,
        this: ViewExpand<E, C, ReadWrite>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewExpand<E, C, ReadWrite> {
        this.__expand_slice_mut_method(scope, pos, size)
    }

    pub fn __expand_slice_mut_unchecked(
        scope: &Scope,
        this: ViewExpand<E, C, ReadWrite>,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewExpand<E, C, ReadWrite> {
        this.__expand_slice_mut_unchecked_method(scope, pos, size)
    }
}

#[cube]
impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadWrite> {}

impl<E: CubePrimitive, C: Coordinates + 'static> ViewExpand<E, C, ReadWrite> {
    pub fn __expand_slice_mut_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewExpand<E, C, ReadWrite> {
        self.slice_mut(scope, pos, size, true)
    }

    pub fn __expand_slice_mut_unchecked_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> ViewExpand<E, C, ReadWrite> {
        self.slice_mut(scope, pos, size, false)
    }

    fn slice_mut(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
        checked: bool,
    ) -> ViewExpand<E, C, ReadWrite> {
        let shape = self.__expand_shape_method(scope);
        let pos = C::__expand_min(scope, pos, shape.clone_unchecked());
        let max_size = C::__expand_sub(scope, shape, pos.clone_unchecked());
        let size = C::__expand_min(scope, size, max_size);
        let layout = SliceLayout::__expand_new(scope, pos, size, checked);
        self.clone().__expand_view_mut_method(scope, layout.into())
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: SliceVisibility> View<E, C, IO> {
    ///.Execute a TMA load into shared memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    pub fn tensor_map_load(
        &self,
        _barrier: &Barrier,
        _shared_memory: &mut [E],
        _pos: C,
    ) -> View<E, C, ReadWrite> {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ViewExpand<E, C, IO> {
    pub fn __expand_tensor_map_load_method(
        &self,
        scope: &Scope,
        barrier: &NativeExpand<Barrier>,
        shared_memory: &mut SliceExpand<E>,
        pos: C::ExpandType,
    ) {
        self.inner
            .read()
            .__expand_tensor_map_load_method(scope, barrier, shared_memory, pos)
    }
}

impl<E: CubePrimitive, C: Coordinates> View<E, C, ReadWrite> {
    ///.Execute a TMA store into global memory, if the underlying storage supports it.
    /// Panics if it's unsupported.
    pub fn tensor_map_store(&self, _shared_memory: &[E], _pos: C) -> View<E, C, ReadWrite> {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates> ViewExpand<E, C, ReadWrite> {
    pub fn __expand_tensor_map_store_method(
        &self,
        scope: &Scope,
        shared_memory: &SliceExpand<E>,
        pos: C::ExpandType,
    ) {
        self.inner
            .write()
            .__expand_tensor_map_store_method(scope, shared_memory, pos)
    }
}

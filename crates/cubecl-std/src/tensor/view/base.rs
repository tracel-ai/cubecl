use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};

use crate::tensor::{
    ViewOperations, ViewOperationsExpand, ViewOperationsMut, ViewOperationsMutExpand, VirtualView,
    VirtualViewMut,
    layout::{Coordinates, Layout, VirtualLayout, VirtualLayoutExpand, slice::SliceLayout},
};

/// A conceptual view of an underlying linear storage.
/// Allows abstract indexing in multiple dimensions, without having to know the data layout or
/// location.
#[derive(Clone)]
pub struct View<E: CubePrimitive, C: Coordinates, IO: Clone = ReadOnly> {
    _layout: PhantomData<C>,
    _ty: PhantomData<(E, IO)>,
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> Copy for View<E, C, IO> {}

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

/// Expand type of [TensorView]
#[derive(Clone)]
pub struct ViewExpand<E: CubePrimitive, C: Coordinates, IO: Clone = ReadOnly> {
    pub(super) inner: ViewType<E, C>,
    pub(super) _io: PhantomData<IO>,
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeType for View<E, C, IO> {
    type ExpandType = ViewExpand<E, C, IO>;
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> IntoMut for ViewExpand<E, C, IO> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeDebug for ViewExpand<E, C, IO> {}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadOnly> {
    /// Create a new tensor view from an underlying concrete storage and a layout to map it into
    /// the target coordinate space
    #[allow(unused_variables)]
    pub fn new<V: ViewOperations<E, S>, S: Coordinates>(
        view: &V,
        layout: impl Into<VirtualLayout<C, S>>,
    ) -> Self {
        View {
            _layout: PhantomData,
            _ty: PhantomData,
        }
    }

    /// Expand function for [TensorView::new]
    pub fn __expand_new<V: ViewOperations<E, S> + 'static, S: Coordinates + 'static>(
        scope: &mut Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewExpand<E, C, ReadOnly> {
        let virt = VirtualView::<E, C, S, V>::__expand_new(scope, view, layout);
        ViewExpand::<E, C, ReadOnly> {
            inner: ViewType::Read(Arc::new(virt)),
            _io: PhantomData,
        }
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
        scope: &mut Scope,
        this: ViewExpand<E, C, IO>,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadOnly> {
        this.__expand_view_method(scope, layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> ViewExpand<E, C, IO> {
    pub fn __expand_view_method<T: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadOnly> {
        View::__expand_new::<View<E, C, IO>, C>(scope, self, layout)
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
        scope: &mut Scope,
        this: ViewExpand<E, C, ReadWrite>,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadWrite> {
        this.__expand_view_mut_method(scope, layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> ViewExpand<E, C, ReadWrite> {
    pub fn __expand_view_mut_method<T: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<T, C>,
    ) -> ViewExpand<E, T, ReadWrite> {
        View::__expand_new_mut::<View<E, C, ReadWrite>, C>(scope, self, layout)
    }
}

impl<E: CubePrimitive, C: Coordinates + 'static> View<E, C, ReadWrite> {
    /// Create a new mutable tensor view from an underlying concrete storage and a layout to map it
    /// into the target coordinate space
    pub fn new_mut<V: ViewOperationsMut<E, S>, S: Coordinates>(
        _view: &mut V,
        _layout: impl Into<VirtualLayout<C, S>>,
    ) -> View<E, C, ReadWrite> {
        View {
            _ty: PhantomData,
            _layout: PhantomData,
        }
    }

    /// Expand function for [TensorView::new_mut]
    pub fn __expand_new_mut<V: ViewOperationsMut<E, S> + 'static, S: Coordinates + 'static>(
        scope: &mut Scope,
        view: V::ExpandType,
        layout: VirtualLayoutExpand<C, S>,
    ) -> ViewExpand<E, C, ReadWrite> {
        let virt = VirtualViewMut::<E, C, S, V>::__expand_new(scope, view, layout);
        ViewExpand::<E, C, ReadWrite> {
            inner: ViewType::ReadWrite(Arc::new(virt)),
            _io: PhantomData,
        }
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> View<E, C, IO> {
    /// Calls [Layout::shape] on the view's layout
    pub fn shape(&self) -> C {
        unexpanded!()
    }

    /// Calls [Layout::is_in_bounds] on the view's layout
    pub fn is_in_bounds(&self, _pos: C) -> bool {
        unexpanded!()
    }

    pub fn __expand_shape(scope: &mut Scope, this: ViewExpand<E, C, IO>) -> C::ExpandType {
        this.__expand_shape_method(scope)
    }

    pub fn __expand_is_in_bounds(
        scope: &mut Scope,
        this: ViewExpand<E, C, IO>,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<bool> {
        this.__expand_is_in_bounds_method(scope, pos)
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ViewExpand<E, C, IO> {
    pub fn __expand_shape_method(self, scope: &mut Scope) -> C::ExpandType {
        self.inner.read().__expand_shape_method(scope)
    }

    pub fn __expand_is_in_bounds_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<bool> {
        self.inner.read().__expand_is_in_bounds_method(scope, pos)
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates, IO: Clone> View<E, C, IO> {
    /// Read a line at `pos`. The layout handles translation into a concrete index.
    pub fn read(&self, pos: C) -> E {
        unexpanded!()
    }

    /// Read a line at `pos`. The layout handles translation into a concrete index.
    /// Reading is done unchecked
    pub fn read_unchecked(&self, pos: C) -> E {
        unexpanded!()
    }

    /// Read a line at `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn read_checked(&self, pos: C) -> E {
        unexpanded!()
    }

    pub fn to_linear_slice(&self) -> Slice<E, ReadOnly> {
        unexpanded!()
    }

    pub fn line_size(&self) -> u32 {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> ViewExpand<E, C, IO> {
    /// Expand method for [TensorView::read]
    pub fn __expand_read_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<E> {
        self.inner.read().__expand_read_method(scope, pos)
    }

    /// Expand method for [TensorView::read_unchecked]
    pub fn __expand_read_unchecked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<E> {
        self.inner.read().__expand_read_unchecked_method(scope, pos)
    }

    /// Expand method for [TensorView::read_checked]
    pub fn __expand_read_checked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<E> {
        self.inner.read().__expand_read_checked_method(scope, pos)
    }

    /// Expand method for [TensorView::line_size]
    pub fn __expand_line_size_method(self, _scope: &mut Scope) -> u32 {
        self.inner.read().line_size()
    }

    pub fn line_size(&self) -> u32 {
        self.inner.read().line_size()
    }

    pub fn __expand_to_linear_slice_method(self, scope: &mut Scope) -> SliceExpand<E, ReadOnly> {
        let shape = self.inner.read().__expand_shape_method(scope);
        let origin = C::__expand_origin(scope, shape.clone());
        self.inner
            .read()
            .__expand_to_linear_slice_method(scope, origin, shape)
    }

    pub(super) fn __expand_to_linear_slice_inner_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        size: C::ExpandType,
    ) -> SliceExpand<E, ReadOnly> {
        self.inner
            .read()
            .__expand_to_linear_slice_method(scope, pos, size)
    }
}

#[cube]
impl<E: CubePrimitive, C: Coordinates + 'static, IO: Clone + 'static> View<E, C, IO> {
    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    pub fn slice(&self, pos: C, size: C) -> View<E, C, ReadOnly> {
        let layout = SliceLayout::new(pos, size);
        self.view(layout)
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates> View<E, C, ReadWrite> {
    /// Write a line to `pos`. The layout handles translation into a concrete index.
    pub fn write(&self, pos: C, value: E) {
        unexpanded!()
    }

    /// Write a line to `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn write_checked(&self, pos: C, value: E) {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates> ViewExpand<E, C, ReadWrite> {
    /// Expand method for [TensorView::write]
    pub fn __expand_write_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<E>,
    ) {
        self.inner.write().__expand_write_method(scope, pos, value);
    }

    /// Expand method for [TensorView::write_checked]
    pub fn __expand_write_checked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<E>,
    ) {
        self.inner
            .write()
            .__expand_write_checked_method(scope, pos, value);
    }
}

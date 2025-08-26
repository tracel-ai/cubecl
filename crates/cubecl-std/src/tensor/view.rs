use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, io::read_masked, unexpanded};

use crate::tensor::{
    layout::{Coordinates, VirtualLayout, VirtualLayoutExpand},
    r#virtual::{Read, ReadWrite},
};

/// A conceptual view of an underlying linear storage.
/// Allows abstract indexing in multiple dimensions, without having to know the data layout or
/// location.
#[derive(Clone)]
pub struct TensorView<E: CubePrimitive, C: Coordinates, IO: Clone = Read> {
    pub layout: VirtualLayout<C>,
    _list: PhantomData<(E, IO)>,
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> Copy for TensorView<E, C, IO> {}

#[derive(Clone)]
enum ListType<E: CubePrimitive> {
    Read(Arc<dyn ListExpand<Line<E>>>),
    ReadWrite(Arc<dyn ListMutExpand<Line<E>>>),
}

impl<E: CubePrimitive> ListType<E> {
    /// Dereference in read mode
    pub fn read(&self) -> &dyn ListExpand<Line<E>> {
        match self {
            ListType::Read(list) => &**list,
            ListType::ReadWrite(list) => &**list,
        }
    }

    /// Dereference in write mode
    pub fn write(&self) -> &dyn ListMutExpand<Line<E>> {
        match self {
            ListType::Read(_) => panic!("Can't write to readonly list"),
            ListType::ReadWrite(list) => &**list,
        }
    }
}

/// Expand type of [TensorView]
#[derive(Clone)]
pub struct TensorViewExpand<E: CubePrimitive, C: Coordinates, IO: Clone = Read> {
    tensor: ListType<E>,
    layout: VirtualLayoutExpand<C>,
    _io: PhantomData<IO>,
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeType for TensorView<E, C, IO> {
    type ExpandType = TensorViewExpand<E, C, IO>;
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> IntoMut for TensorViewExpand<E, C, IO> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeDebug for TensorViewExpand<E, C, IO> {}

impl<E: CubePrimitive, C: Coordinates> TensorView<E, C, Read> {
    /// Create a new tensor view from an underlying concrete storage and a layout to map it into
    /// the target coordinate space
    #[allow(unused_variables)]
    pub fn new<T>(tensor: T, layout: VirtualLayout<C>) -> Self
    where
        T: List<Line<E>> + CubeType,
        T::ExpandType: ListExpand<Line<E>> + 'static,
    {
        TensorView {
            layout,
            _list: PhantomData,
        }
    }

    /// Expand function for [TensorView::new]
    pub fn __expand_new<T>(
        _scope: &mut Scope,
        tensor: T::ExpandType,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, Read>
    where
        T: List<Line<E>> + CubeType,
        T::ExpandType: ListExpand<Line<E>> + 'static,
    {
        TensorViewExpand::<E, C, Read> {
            tensor: ListType::Read(Arc::new(tensor)),
            layout,
            _io: PhantomData,
        }
    }
}

impl<E: CubePrimitive, C: Coordinates> TensorView<E, C, ReadWrite> {
    /// Create a new mutable tensor view from an underlying concrete storage and a layout to map it
    /// into the target coordinate space
    pub fn new_mut<T>(_tensor: T, layout: VirtualLayout<C>) -> TensorView<E, C, ReadWrite>
    where
        T: ListMut<Line<E>> + CubeType,
        T::ExpandType: ListMutExpand<Line<E>> + 'static,
    {
        TensorView {
            layout,
            _list: PhantomData,
        }
    }

    /// Expand function for [TensorView::new_mut]
    pub fn __expand_new_mut<T>(
        _scope: &mut Scope,
        tensor: T::ExpandType,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, ReadWrite>
    where
        T: ListMut<Line<E>> + CubeType,
        T::ExpandType: ListMutExpand<Line<E>> + 'static,
    {
        TensorViewExpand::<E, C, ReadWrite> {
            tensor: ListType::ReadWrite(Arc::new(tensor)),
            layout,
            _io: PhantomData,
        }
    }
}

#[cube]
impl<E: CubePrimitive, C: Coordinates, IO: Clone> TensorView<E, C, IO> {
    /// Calls [Layout::to_linear_pos] on the view's layout
    #[allow(unused)]
    pub fn to_linear_pos(&self, pos: C) -> u32 {
        self.layout.to_linear_pos(pos)
    }

    /// Calls [Layout::to_linear_pos_checked] on the view's layout
    #[allow(unused)]
    pub fn to_linear_pos_checked(&self, pos: C) -> (u32, bool) {
        self.layout.to_linear_pos_checked(pos)
    }

    /// Calls [Layout::shape] on the view's layout
    pub fn shape(&self) -> C {
        self.layout.shape()
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates, IO: Clone> TensorView<E, C, IO> {
    /// Read a line at `pos`. The layout handles translation into a concrete index.
    pub fn read(&self, pos: C) -> Line<E> {
        unexpanded!()
    }

    /// Read a line at `pos`. The layout handles translation into a concrete index.
    /// Reading is done unchecked
    pub fn read_unchecked(&self, pos: C) -> Line<E> {
        unexpanded!()
    }

    /// Read a line at `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn read_checked(&self, pos: C) -> Line<E> {
        unexpanded!()
    }

    /// Create a slice starting from `pos`, with `size`.
    /// The layout handles translation into concrete indices.
    pub fn slice(&self, pos: C, size: u32) -> Slice<Line<E>, ReadOnly> {
        unexpanded!()
    }

    pub fn line_size(&self) -> u32 {
        unexpanded!()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates, IO: Clone> TensorViewExpand<E, C, IO> {
    /// Expand method for [TensorView::read]
    pub fn __expand_read_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<Line<E>> {
        let read_pos = self.clone().__expand_to_linear_pos_method(scope, pos);
        self.tensor.read().__expand_read_method(scope, read_pos)
    }

    /// Expand method for [TensorView::read_unchecked]
    pub fn __expand_read_unchecked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<Line<E>> {
        let read_pos = self.clone().__expand_to_linear_pos_method(scope, pos);
        self.tensor
            .read()
            .__expand_read_unchecked_method(scope, read_pos)
    }

    /// Expand method for [TensorView::read_checked]
    pub fn __expand_read_checked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<Line<E>> {
        let slice = self.clone().tensor.read().__expand_to_slice_method(scope);
        let (read_pos, in_bounds) = self.__expand_to_linear_pos_checked_method(scope, pos);
        let zero = Line::__expand_cast_from(scope, 0.into());
        read_masked::expand::<Line<E>>(scope, in_bounds, slice, read_pos, zero)
    }

    /// Expand method for [TensorView::slice]
    pub fn __expand_slice_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        size: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        let start = self.clone().__expand_to_linear_pos_method(scope, pos);
        let end = add::expand(scope, start.clone(), size);
        self.tensor.read().__expand_slice_method(scope, start, end)
    }

    /// Expand method for [TensorView::line_size]
    pub fn __expand_line_size_method(self, scope: &mut Scope) -> u32 {
        self.tensor.read().__expand_line_size_method(scope)
    }

    pub fn line_size(&self) -> u32 {
        self.tensor.read().line_size()
    }

    pub fn __expand_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        self.tensor.read().__expand_len_method(scope)
    }
}

#[allow(unused_variables)]
impl<E: CubePrimitive, C: Coordinates> TensorView<E, C, ReadWrite> {
    /// Write a line to `pos`. The layout handles translation into a concrete index.
    pub fn write(&self, pos: C, value: Line<E>) {
        unexpanded!()
    }

    /// Write a line to `pos` if it's in bounds. The layout handles translation into a concrete index.
    pub fn write_checked(&self, pos: C, value: Line<E>) {
        unexpanded!()
    }
}

impl<E: CubePrimitive, C: Coordinates> TensorViewExpand<E, C, ReadWrite> {
    /// Expand method for [TensorView::write]
    pub fn __expand_write_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<Line<E>>,
    ) {
        let write_pos = self.clone().__expand_to_linear_pos_method(scope, pos);
        self.tensor
            .write()
            .__expand_write_method(scope, write_pos, value)
    }

    /// Expand method for [TensorView::write_checked]
    pub fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<Line<E>>,
    ) {
        let (write_pos, in_bounds) = self
            .clone()
            .__expand_to_linear_pos_checked_method(scope, pos);
        if_expand(scope, in_bounds.into(), |scope| {
            self.tensor
                .write()
                .__expand_write_method(scope, write_pos, value)
        });
    }
}

pub trait AsView<E: CubePrimitive> {
    #[allow(unused)]
    fn view<C: Coordinates>(&self, layout: VirtualLayout<C>) -> TensorView<E, C, Read> {
        unexpanded!()
    }
}

pub trait AsViewExpand<E: CubePrimitive> {
    fn __expand_view_method<C: Coordinates>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, Read>;
}

pub trait AsViewMut<E: CubePrimitive> {
    #[allow(unused)]
    fn view_mut<C: Coordinates>(
        &mut self,
        layout: VirtualLayout<C>,
    ) -> TensorView<E, C, ReadWrite> {
        unexpanded!()
    }
}

pub trait AsViewMutExpand<E: CubePrimitive> {
    fn __expand_view_mut_method<C: Coordinates>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, ReadWrite>;
}

mod as_view {
    use super::*;

    impl<E: CubePrimitive, L> AsView<E> for L
    where
        L: List<Line<E>> + CubeType<ExpandType = ExpandElementTyped<L>> + 'static,
        L::ExpandType: ListExpand<Line<E>>,
    {
    }
    impl<E: CubePrimitive, L> AsViewExpand<E> for ExpandElementTyped<L>
    where
        L: List<Line<E>> + CubeType<ExpandType = ExpandElementTyped<L>> + 'static,
        L::ExpandType: ListExpand<Line<E>>,
    {
        fn __expand_view_method<C: Coordinates>(
            self,
            scope: &mut Scope,
            layout: VirtualLayoutExpand<C>,
        ) -> super::TensorViewExpand<E, C, Read> {
            TensorView::__expand_new::<L>(scope, self, layout)
        }
    }

    impl<E: CubePrimitive, L> AsViewMut<E> for L
    where
        L: ListMut<Line<E>> + CubeType<ExpandType = ExpandElementTyped<L>> + 'static,
        L::ExpandType: ListMutExpand<Line<E>>,
    {
    }
    impl<E: CubePrimitive, L> AsViewMutExpand<E> for ExpandElementTyped<L>
    where
        L: ListMut<Line<E>> + CubeType<ExpandType = ExpandElementTyped<L>> + 'static,
        L::ExpandType: ListMutExpand<Line<E>>,
    {
        fn __expand_view_mut_method<C: Coordinates>(
            self,
            scope: &mut Scope,
            layout: VirtualLayoutExpand<C>,
        ) -> super::TensorViewExpand<E, C, ReadWrite> {
            TensorView::__expand_new_mut::<L>(scope, self, layout)
        }
    }
}

mod idx {
    use super::*;

    impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeIndex<C> for TensorView<E, C, IO> {
        type Output = Line<E>;
    }

    impl<E: CubePrimitive, C: Coordinates, IO: Clone> CubeIndexExpand<C>
        for TensorViewExpand<E, C, IO>
    {
        type Output = <Line<E> as CubeType>::ExpandType;

        fn expand_index(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
            self.__expand_read_method(scope, index)
        }

        fn expand_index_unchecked(self, scope: &mut Scope, index: C::ExpandType) -> Self::Output {
            self.__expand_read_unchecked_method(scope, index)
        }
    }

    impl<E: CubePrimitive, C: Coordinates> CubeIndexMut<C> for TensorView<E, C, ReadWrite> {}
    impl<E: CubePrimitive, C: Coordinates> CubeIndexMutExpand<C> for TensorViewExpand<E, C, ReadWrite> {
        type Output = <Line<E> as CubeType>::ExpandType;

        fn expand_index_mut(self, scope: &mut Scope, index: C::ExpandType, value: Self::Output) {
            self.__expand_write_method(scope, index, value)
        }
    }
}

pub mod launch {
    use crate::tensor::{
        layout::{Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand},
        r#virtual::ReadWrite,
    };

    use super::*;
    use std::{
        marker::PhantomData,
        ops::{Deref, DerefMut},
    };

    /// Launchable tensor view for ease of use.
    #[derive(Clone)]
    pub struct TensorViewTyped<E: CubePrimitive, L: Layout, IO: Clone = Read> {
        _ty: PhantomData<(E, L, IO)>,
    }

    impl<E: CubePrimitive, L: Layout, IO: Clone> CubeType for TensorViewTyped<E, L, IO> {
        type ExpandType = TensorViewExpand<E, L::Coordinates, IO>;
    }

    impl<E: CubePrimitive, L: Layout, IO: Clone> Deref for TensorViewTyped<E, L, IO> {
        type Target = TensorView<E, L::Coordinates, IO>;

        fn deref(&self) -> &Self::Target {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive, L: Layout> DerefMut for TensorViewTyped<E, L, ReadWrite> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            unexpanded!()
        }
    }

    pub struct TensorViewTypedLaunch<'a, L: Layout + CubeLaunch, R: Runtime> {
        buffer: ArrayArg<'a, R>,
        layout: L::RuntimeArg<'a, R>,
    }
    impl<'a, L: Layout + CubeLaunch, R: Runtime> TensorViewTypedLaunch<'a, L, R> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(buffer: ArrayArg<'a, R>, layout: L::RuntimeArg<'a, R>) -> Self {
            Self { buffer, layout }
        }
    }
    impl<'a, L: Layout + CubeLaunch, R: Runtime> ArgSettings<R> for TensorViewTypedLaunch<'a, L, R> {
        fn register(&self, launcher: &mut cubecl::prelude::KernelLauncher<R>) {
            self.buffer.register(launcher);
            self.layout.register(launcher);
        }
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    #[serde(bound(serialize = "", deserialize = ""))]
    pub struct TensorViewTypedCompilationArg<L: Layout + CubeLaunch> {
        buffer: ArrayCompilationArg,
        layout: L::CompilationArg,
    }
    impl<L: Layout + CubeLaunch> Clone for TensorViewTypedCompilationArg<L> {
        fn clone(&self) -> Self {
            Self {
                buffer: self.buffer.clone(),
                layout: self.layout.clone(),
            }
        }
    }
    impl<L: Layout + CubeLaunch> CompilationArg for TensorViewTypedCompilationArg<L> {}

    impl<L: Layout + CubeLaunch> core::hash::Hash for TensorViewTypedCompilationArg<L> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.buffer.hash(state);
            self.layout.hash(state);
        }
    }
    impl<L: Layout + CubeLaunch> PartialEq for TensorViewTypedCompilationArg<L> {
        fn eq(&self, other: &Self) -> bool {
            self.buffer.eq(&other.buffer) && self.layout.eq(&other.layout)
        }
    }
    impl<L: Layout + CubeLaunch> core::fmt::Debug for TensorViewTypedCompilationArg<L> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct(stringify!(TensorViewTyped))
                .field("buffer", &self.buffer)
                .field("layout", &self.layout)
                .finish()
        }
    }
    impl<L: Layout + CubeLaunch> Eq for TensorViewTypedCompilationArg<L> {}

    impl<E: CubePrimitive, L: Layout + CubeLaunch, IO: Clone + Send + Sync + 'static> LaunchArg
        for TensorViewTyped<E, L, IO>
    where
        L: VirtualLayoutOperations<L::Coordinates>,
        L::ExpandType: VirtualLayoutOperationsExpand<L::Coordinates>,
    {
        type RuntimeArg<'a, R: Runtime> = TensorViewTypedLaunch<'a, L, R>;

        fn compilation_arg<'a, R: Runtime>(
            runtime_arg: &Self::RuntimeArg<'a, R>,
        ) -> Self::CompilationArg {
            TensorViewTypedCompilationArg {
                buffer: <Array<Line<E>> as LaunchArg>::compilation_arg(&runtime_arg.buffer),
                layout: L::compilation_arg(&runtime_arg.layout),
            }
        }
    }
    impl<E: CubePrimitive, L: Layout + CubeLaunch, IO: Clone + Send + Sync + 'static>
        LaunchArgExpand for TensorViewTyped<E, L, IO>
    where
        L: VirtualLayoutOperations<L::Coordinates>,
        L::ExpandType: VirtualLayoutOperationsExpand<L::Coordinates>,
    {
        type CompilationArg = TensorViewTypedCompilationArg<L>;
        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            let buffer = <Array<Line<E>> as LaunchArgExpand>::expand(&arg.buffer, builder);
            let layout = L::expand(&arg.layout, builder);
            TensorViewExpand::<E, L::Coordinates, IO> {
                tensor: ListType::Read(Arc::new(buffer)),
                layout: VirtualLayoutExpand {
                    state: Arc::new(layout),
                },
                _io: PhantomData,
            }
        }
        fn expand_output(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            let buffer = <Array<Line<E>> as LaunchArgExpand>::expand_output(&arg.buffer, builder);
            let layout = L::expand_output(&arg.layout, builder);
            TensorViewExpand::<E, L::Coordinates, IO> {
                tensor: ListType::ReadWrite(Arc::new(buffer)),
                layout: VirtualLayoutExpand {
                    state: Arc::new(layout),
                },
                _io: PhantomData,
            }
        }
    }
}

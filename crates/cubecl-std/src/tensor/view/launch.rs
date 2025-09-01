use cubecl_core::{prelude::*, unexpanded};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::tensor::{
    View, ViewExpand, VirtualViewExpand, VirtualViewMutExpand,
    layout::{
        Coords1d, Layout, VirtualLayoutExpand, VirtualLayoutOperations,
        VirtualLayoutOperationsExpand,
    },
    view::ViewType,
    r#virtual::{Read, ReadWrite},
};

/// Launchable tensor view for ease of use.
#[derive(Clone)]
pub struct TypedView<E: CubePrimitive, L: Layout, IO: Clone = Read> {
    _ty: PhantomData<(E, L, IO)>,
}

impl<E: CubePrimitive, L: Layout, IO: Clone> CubeType for TypedView<E, L, IO> {
    type ExpandType = ViewExpand<E, L::Coordinates, IO>;
}

impl<E: CubePrimitive, L: Layout, IO: Clone> Deref for TypedView<E, L, IO> {
    type Target = View<E, L::Coordinates, IO>;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<E: CubePrimitive, L: Layout> DerefMut for TypedView<E, L, ReadWrite> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

pub struct TypedViewLaunch<'a, L: Layout<SourceCoordinates = Coords1d> + CubeLaunch, R: Runtime> {
    buffer: ArrayArg<'a, R>,
    layout: L::RuntimeArg<'a, R>,
}
impl<'a, L: Layout<SourceCoordinates = Coords1d> + CubeLaunch, R: Runtime>
    TypedViewLaunch<'a, L, R>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(buffer: ArrayArg<'a, R>, layout: L::RuntimeArg<'a, R>) -> Self {
        Self { buffer, layout }
    }
}
impl<'a, L: Layout<SourceCoordinates = Coords1d> + CubeLaunch, R: Runtime> ArgSettings<R>
    for TypedViewLaunch<'a, L, R>
{
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        self.buffer.register(launcher);
        self.layout.register(launcher);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TypedViewCompilationArg<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> {
    buffer: ArrayCompilationArg,
    layout: L::CompilationArg,
}
impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> Clone for TypedViewCompilationArg<L> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            layout: self.layout.clone(),
        }
    }
}
impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> CompilationArg
    for TypedViewCompilationArg<L>
{
}

impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> core::hash::Hash
    for TypedViewCompilationArg<L>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
        self.layout.hash(state);
    }
}
impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> PartialEq
    for TypedViewCompilationArg<L>
{
    fn eq(&self, other: &Self) -> bool {
        self.buffer.eq(&other.buffer) && self.layout.eq(&other.layout)
    }
}
impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> core::fmt::Debug
    for TypedViewCompilationArg<L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(stringify!(TensorViewTyped))
            .field("buffer", &self.buffer)
            .field("layout", &self.layout)
            .finish()
    }
}
impl<L: Layout<SourceCoordinates = Coords1d> + CubeLaunch> Eq for TypedViewCompilationArg<L> {}

impl<
    E: CubePrimitive,
    L: Layout<SourceCoordinates = Coords1d> + CubeLaunch,
    IO: Clone + Send + Sync + 'static,
> LaunchArg for TypedView<E, L, IO>
where
    L: VirtualLayoutOperations<L::Coordinates, L::SourceCoordinates>,
    L::ExpandType: VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates>,
{
    type RuntimeArg<'a, R: Runtime> = TypedViewLaunch<'a, L, R>;

    fn compilation_arg<'a, R: Runtime>(
        runtime_arg: &Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
        TypedViewCompilationArg {
            buffer: <Array<Line<E>> as LaunchArg>::compilation_arg(&runtime_arg.buffer),
            layout: L::compilation_arg(&runtime_arg.layout),
        }
    }
}
impl<
    E: CubePrimitive,
    L: Layout<SourceCoordinates = Coords1d> + CubeLaunch,
    IO: Clone + Send + Sync + 'static,
> LaunchArgExpand for TypedView<E, L, IO>
where
    L: VirtualLayoutOperations<L::Coordinates, L::SourceCoordinates>,
    L::ExpandType: VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates>,
{
    type CompilationArg = TypedViewCompilationArg<L>;
    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let buffer = <Array<E> as LaunchArgExpand>::expand(&arg.buffer, builder);
        let layout = VirtualLayoutExpand::new::<L>(L::expand(&arg.layout, builder));
        let view = VirtualViewExpand::<E, L::Coordinates, Coords1d, Array<E>>::new(buffer, layout);
        ViewExpand::<E, L::Coordinates, IO> {
            inner: ViewType::Read(Arc::new(view)),
            _io: PhantomData,
        }
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let buffer = <Array<E> as LaunchArgExpand>::expand_output(&arg.buffer, builder);
        let layout = VirtualLayoutExpand::new::<L>(L::expand_output(&arg.layout, builder));
        let view =
            VirtualViewMutExpand::<E, L::Coordinates, Coords1d, Array<E>>::new(buffer, layout);
        ViewExpand::<E, L::Coordinates, IO> {
            inner: ViewType::ReadWrite(Arc::new(view)),
            _io: PhantomData,
        }
    }
}

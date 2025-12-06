use cubecl_core::{prelude::*, unexpanded};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::tensor::{
    View, ViewExpand, ViewOperationsMut, VirtualViewMut, VirtualViewMutExpand,
    layout::{Coordinates, Coords1d, Layout, VirtualLayoutExpand, VirtualLayoutOperationsExpand},
    view::ViewType,
};

/// Launchable tensor view for ease of use.
#[derive(Clone)]
pub struct TypedView<E: CubePrimitive, L: LaunchLayout, IO: SliceVisibility = ReadOnly> {
    _ty: PhantomData<(E, L, IO)>,
}

impl<E: CubePrimitive, L: LaunchLayout, IO: SliceVisibility> CubeType for TypedView<E, L, IO> {
    type ExpandType = ViewExpand<E, L::Coordinates, IO>;
}

impl<E: CubePrimitive, L: LaunchLayout, IO: SliceVisibility> Deref for TypedView<E, L, IO> {
    type Target = View<E, L::Coordinates, IO>;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<E: CubePrimitive, L: LaunchLayout> DerefMut for TypedView<E, L, ReadWrite> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

pub struct TypedViewLaunch<'a, L: LaunchLayout<SourceCoordinates = Coords1d>, R: Runtime> {
    buffer: ArrayArg<'a, R>,
    layout: L::RuntimeArg<'a, R>,
}
impl<'a, L: LaunchLayout<SourceCoordinates = Coords1d>, R: Runtime> TypedViewLaunch<'a, L, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(buffer: ArrayArg<'a, R>, layout: L::RuntimeArg<'a, R>) -> Self {
        Self { buffer, layout }
    }
}
impl<'a, L: LaunchLayout<SourceCoordinates = Coords1d>, R: Runtime> ArgSettings<R>
    for TypedViewLaunch<'a, L, R>
{
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        self.buffer.register(launcher);
        self.layout.register(launcher);
    }
}

pub struct TypedViewCompilationArg<L: LaunchLayout<SourceCoordinates = Coords1d>> {
    buffer: ArrayCompilationArg,
    layout: L::CompilationArg,
}
impl<L: LaunchLayout<SourceCoordinates = Coords1d>> Clone for TypedViewCompilationArg<L> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            layout: self.layout.clone(),
        }
    }
}
impl<L: LaunchLayout<SourceCoordinates = Coords1d>> CompilationArg for TypedViewCompilationArg<L> {}

impl<L: LaunchLayout<SourceCoordinates = Coords1d>> core::hash::Hash
    for TypedViewCompilationArg<L>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
        self.layout.hash(state);
    }
}
impl<L: LaunchLayout<SourceCoordinates = Coords1d>> PartialEq for TypedViewCompilationArg<L> {
    fn eq(&self, other: &Self) -> bool {
        self.buffer.eq(&other.buffer) && self.layout.eq(&other.layout)
    }
}
impl<L: LaunchLayout<SourceCoordinates = Coords1d>> core::fmt::Debug
    for TypedViewCompilationArg<L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(stringify!(TensorViewTyped))
            .field("buffer", &self.buffer)
            .field("layout", &self.layout)
            .finish()
    }
}
impl<L: LaunchLayout<SourceCoordinates = Coords1d>> Eq for TypedViewCompilationArg<L> {}

impl<E: CubePrimitive, L: LaunchLayout<SourceCoordinates = Coords1d>, IO: SliceVisibility> LaunchArg
    for TypedView<E, L, IO>
{
    type RuntimeArg<'a, R: Runtime> = TypedViewLaunch<'a, L, R>;
    type CompilationArg = TypedViewCompilationArg<L>;

    fn compilation_arg<'a, R: Runtime>(
        runtime_arg: &Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
        TypedViewCompilationArg {
            buffer: <Array<Line<E>> as LaunchArg>::compilation_arg(&runtime_arg.buffer),
            layout: L::compilation_arg(&runtime_arg.layout),
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let buffer = <Array<E> as LaunchArg>::expand(&arg.buffer, builder);
        L::apply::<E, Array<E>, IO>(L::expand(&arg.layout, builder), buffer)
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let buffer = <Array<E> as LaunchArg>::expand_output(&arg.buffer, builder);
        L::apply::<E, Array<E>, IO>(L::expand_output(&arg.layout, builder), buffer)
    }
}

mod seal {
    pub trait Sealed {}
}

pub trait LaunchLayout: LaunchArg + seal::Sealed {
    type SourceCoordinates: Coordinates;
    type Coordinates: Coordinates;

    fn apply<
        E: CubePrimitive,
        V: ViewOperationsMut<E, Self::SourceCoordinates> + 'static,
        IO: SliceVisibility,
    >(
        value: <Self as CubeType>::ExpandType,
        view: V::ExpandType,
    ) -> ViewExpand<E, Self::Coordinates, IO>;
}

// These unfortunately need to be manually implemented due to the dependencies of each layout on
// the coordinates of the next. Just stick with two layouts for now and add more implementations as
// needed.

impl<
    L: Layout
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates>>
        + LaunchArg,
> seal::Sealed for L
{
}
impl<
    L: Layout
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates>>
        + LaunchArg,
> LaunchLayout for L
{
    type SourceCoordinates = L::SourceCoordinates;
    type Coordinates = L::Coordinates;

    fn apply<
        E: CubePrimitive,
        V: ViewOperationsMut<E, Self::SourceCoordinates> + 'static,
        IO: SliceVisibility,
    >(
        value: L::ExpandType,
        view: V::ExpandType,
    ) -> ViewExpand<E, Self::Coordinates, IO> {
        let l0 = value;
        let l0 = VirtualLayoutExpand::new::<L::ExpandType>(l0);
        let view =
            VirtualViewMutExpand::<E, L::Coordinates, L::SourceCoordinates, V>::new(view, l0);
        ViewExpand::<E, L::Coordinates, IO> {
            inner: ViewType::ReadWrite(Arc::new(view)),
            _io: PhantomData,
        }
    }
}

impl<
    L0: Layout
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L0::Coordinates, L0::SourceCoordinates>>
        + LaunchArg,
    L1: Layout<SourceCoordinates = L0::Coordinates>
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L1::Coordinates, L1::SourceCoordinates>>
        + LaunchArg,
> seal::Sealed for (L0, L1)
{
}
impl<
    L0: Layout
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L0::Coordinates, L0::SourceCoordinates>>
        + LaunchArg,
    L1: Layout<SourceCoordinates = L0::Coordinates>
        + CubeType<ExpandType: VirtualLayoutOperationsExpand<L1::Coordinates, L1::SourceCoordinates>>
        + LaunchArg,
> LaunchLayout for (L0, L1)
{
    type SourceCoordinates = L0::SourceCoordinates;
    type Coordinates = L1::Coordinates;

    fn apply<
        E: CubePrimitive,
        V: ViewOperationsMut<E, Self::SourceCoordinates> + 'static,
        IO: SliceVisibility,
    >(
        value: (L0::ExpandType, L1::ExpandType),
        view: V::ExpandType,
    ) -> ViewExpand<E, Self::Coordinates, IO> {
        let (l0, l1) = value;
        let l0 = VirtualLayoutExpand::new::<L0::ExpandType>(l0);
        let view =
            VirtualViewMutExpand::<E, L0::Coordinates, L0::SourceCoordinates, V>::new(view, l0);
        let l1 = VirtualLayoutExpand::new::<L1::ExpandType>(l1);
        let view = VirtualViewMutExpand::<
            E,
            L1::Coordinates,
            L1::SourceCoordinates,
            VirtualViewMut<E, L0::Coordinates, L0::SourceCoordinates, V>,
        >::new(view, l1);
        ViewExpand::<E, L1::Coordinates, IO> {
            inner: ViewType::ReadWrite(Arc::new(view)),
            _io: PhantomData,
        }
    }
}

mod dynamic {
    use cubecl_common::quant::scheme::QuantScheme;

    use crate::{
        quant,
        tensor::{
            VirtualViewExpand,
            layout::{
                VirtualLayout, VirtualLayoutCompilationArg, VirtualLayoutLaunch,
                as_dyn::{
                    IntoDyn, IntoDyn2Layout, IntoDyn2LayoutLaunch, IntoDynLayout,
                    IntoDynLayoutLaunch,
                },
            },
        },
    };

    use super::*;

    pub enum ViewArg<'a, C: Coordinates, R: Runtime> {
        Array(ArrayArg<'a, R>, VirtualLayoutLaunch<'a, C, Coords1d, R>),
        TensorMapTiled(
            TensorMapArg<'a, R, Tiled>,
            VirtualLayoutLaunch<'a, C, Sequence<i32>, R>,
        ),
        TensorMapIm2col(
            TensorMapArg<'a, R, Im2col>,
            VirtualLayoutLaunch<'a, C, (Sequence<i32>, Sequence<i32>), R>,
        ),
        Quantized {
            values: Box<ViewArg<'a, C, R>>,
            scales: Box<ViewArg<'a, C, R>>,
            scheme: QuantScheme,
        },
    }
    impl<'a, C: Coordinates, R: Runtime> ViewArg<'a, C, R> {
        pub fn new<L: Layout<Coordinates = C, SourceCoordinates = Coords1d> + LaunchArg>(
            buffer: ArrayArg<'a, R>,
            layout: L::RuntimeArg<'a, R>,
        ) -> Self {
            ViewArg::Array(buffer, VirtualLayoutLaunch::new::<L>(layout))
        }

        pub fn new_tensor_map_tiled<
            L: Layout<Coordinates = C, SourceCoordinates: IntoDyn> + LaunchArg,
        >(
            buffer: TensorMapArg<'a, R, Tiled>,
            layout: L::RuntimeArg<'a, R>,
        ) -> Self {
            let layout = IntoDynLayoutLaunch::new(layout);
            ViewArg::TensorMapTiled(buffer, VirtualLayoutLaunch::new::<IntoDynLayout<L>>(layout))
        }

        pub fn new_tensor_map_im2col<
            L: Layout<Coordinates = C, SourceCoordinates = (P, O)> + LaunchArg,
            P: IntoDyn,
            O: IntoDyn,
        >(
            buffer: TensorMapArg<'a, R, Im2col>,
            layout: L::RuntimeArg<'a, R>,
        ) -> Self {
            let layout = IntoDyn2LayoutLaunch::new(layout);
            ViewArg::TensorMapIm2col(
                buffer,
                VirtualLayoutLaunch::new::<IntoDyn2Layout<L, P, O>>(layout),
            )
        }

        /// Create a new view arg that dequantizes on read.
        /// The scales layout should take values indices and map them to the corresponding scale.
        pub fn new_quantized(values: Self, scales: Self, scheme: QuantScheme) -> Self {
            Self::Quantized {
                values: Box::new(values),
                scales: Box::new(scales),
                scheme,
            }
        }
    }
    impl<'a, C: Coordinates, R: Runtime> ArgSettings<R> for ViewArg<'a, C, R> {
        fn register(&self, launcher: &mut KernelLauncher<R>) {
            match self {
                ViewArg::Array(buffer, layout) => {
                    buffer.register(launcher);
                    layout.register(launcher);
                }
                ViewArg::TensorMapTiled(buffer, layout) => {
                    buffer.register(launcher);
                    layout.register(launcher);
                }
                ViewArg::TensorMapIm2col(buffer, layout) => {
                    buffer.register(launcher);
                    layout.register(launcher);
                }
                ViewArg::Quantized { values, scales, .. } => {
                    values.register(launcher);
                    scales.register(launcher);
                }
            }
        }
    }
    #[derive(Clone)]
    pub enum ViewCompilationArg<C: Coordinates> {
        Array {
            buffer: ArrayCompilationArg,
            layout: VirtualLayoutCompilationArg<C, Coords1d>,
        },
        TensorMapTiled {
            buffer: TensorMapCompilationArg,
            layout: VirtualLayoutCompilationArg<C, Sequence<i32>>,
        },
        TensorMapIm2col {
            buffer: TensorMapCompilationArg,
            layout: VirtualLayoutCompilationArg<C, (Sequence<i32>, Sequence<i32>)>,
        },
        Quantized {
            values: Box<ViewCompilationArg<C>>,
            scales: Box<ViewCompilationArg<C>>,
            scheme: QuantScheme,
        },
    }

    impl<C: Coordinates + 'static> CompilationArg for ViewCompilationArg<C> {}
    impl<C: Coordinates> Eq for ViewCompilationArg<C> {}
    impl<C: Coordinates> PartialEq for ViewCompilationArg<C> {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (
                    ViewCompilationArg::Array { buffer, layout },
                    ViewCompilationArg::Array {
                        buffer: buffer_other,
                        layout: layout_other,
                    },
                ) => buffer == buffer_other && layout == layout_other,
                (
                    ViewCompilationArg::TensorMapTiled { buffer, layout },
                    ViewCompilationArg::TensorMapTiled {
                        buffer: buffer_other,
                        layout: layout_other,
                    },
                ) => buffer == buffer_other && layout == layout_other,
                (
                    ViewCompilationArg::TensorMapIm2col { buffer, layout },
                    ViewCompilationArg::TensorMapIm2col {
                        buffer: buffer_other,
                        layout: layout_other,
                    },
                ) => buffer == buffer_other && layout == layout_other,
                (
                    ViewCompilationArg::Quantized {
                        values,
                        scales,
                        scheme,
                    },
                    ViewCompilationArg::Quantized {
                        values: values_other,
                        scales: scales_other,
                        scheme: scheme_other,
                    },
                ) => values == values_other && scales == scales_other && scheme == scheme_other,
                _ => false,
            }
        }
    }
    impl<C: Coordinates> core::hash::Hash for ViewCompilationArg<C> {
        fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
            match self {
                ViewCompilationArg::Array { buffer, layout } => {
                    buffer.hash(ra_expand_state);
                    layout.hash(ra_expand_state);
                }
                ViewCompilationArg::TensorMapTiled { buffer, layout } => {
                    buffer.hash(ra_expand_state);
                    layout.hash(ra_expand_state);
                }
                ViewCompilationArg::TensorMapIm2col { buffer, layout } => {
                    buffer.hash(ra_expand_state);
                    layout.hash(ra_expand_state);
                }
                ViewCompilationArg::Quantized {
                    values,
                    scales,
                    scheme,
                } => {
                    values.hash(ra_expand_state);
                    scales.hash(ra_expand_state);
                    scheme.hash(ra_expand_state);
                }
            }
        }
    }
    impl<C: Coordinates> core::fmt::Debug for ViewCompilationArg<C> {
        fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
            match self {
                ViewCompilationArg::Array { buffer, layout } => f
                    .debug_struct("ArrayView")
                    .field("buffer", &buffer)
                    .field("layout", &layout)
                    .finish(),
                ViewCompilationArg::TensorMapTiled { buffer, layout } => f
                    .debug_struct("TensorMapTiledView")
                    .field("buffer", &buffer)
                    .field("layout", &layout)
                    .finish(),
                ViewCompilationArg::TensorMapIm2col { buffer, layout } => f
                    .debug_struct("TensorMapIm2colView")
                    .field("buffer", &buffer)
                    .field("layout", &layout)
                    .finish(),
                ViewCompilationArg::Quantized {
                    values,
                    scales,
                    scheme,
                } => f
                    .debug_struct("QuantizedView")
                    .field("values", &values)
                    .field("scales", &scales)
                    .field("scheme", &scheme)
                    .finish(),
            }
        }
    }

    impl<E: CubePrimitive, C: Coordinates + 'static, IO: SliceVisibility> LaunchArg for View<E, C, IO> {
        type RuntimeArg<'a, R: Runtime> = ViewArg<'a, C, R>;
        type CompilationArg = ViewCompilationArg<C>;

        fn compilation_arg<'a, R: Runtime>(
            runtime_arg: &Self::RuntimeArg<'a, R>,
        ) -> Self::CompilationArg {
            match runtime_arg {
                ViewArg::Array(buffer, layout) => {
                    let buffer = Array::<E>::compilation_arg(buffer);
                    let layout = VirtualLayout::<C, Coords1d>::compilation_arg(layout);
                    ViewCompilationArg::Array { buffer, layout }
                }
                ViewArg::TensorMapTiled(buffer, layout) => {
                    let buffer = TensorMap::<E, Tiled>::compilation_arg(buffer);
                    let layout = VirtualLayout::<C, Sequence<i32>>::compilation_arg(layout);
                    ViewCompilationArg::TensorMapTiled { buffer, layout }
                }
                ViewArg::TensorMapIm2col(buffer, layout) => {
                    let buffer = TensorMap::<E, Im2col>::compilation_arg(buffer);
                    let layout =
                        VirtualLayout::<C, (Sequence<i32>, Sequence<i32>)>::compilation_arg(layout);
                    ViewCompilationArg::TensorMapIm2col { buffer, layout }
                }
                ViewArg::Quantized {
                    values,
                    scales,
                    scheme,
                } => {
                    // Type isn't real, but doesn't matter for compilation arg
                    let values = View::<E, C, IO>::compilation_arg(values);
                    let scales = View::<E, C, IO>::compilation_arg(scales);
                    ViewCompilationArg::Quantized {
                        values: Box::new(values),
                        scales: Box::new(scales),
                        scheme: *scheme,
                    }
                }
            }
        }
        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            match arg {
                ViewCompilationArg::Array { buffer, layout } => {
                    let buffer = Array::<E>::expand(buffer, builder);
                    let layout = VirtualLayout::<C, Coords1d>::expand(layout, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Coords1d, Array<E>>::new(buffer, layout);
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapTiled { buffer, layout } => {
                    let buffer = TensorMap::<E, Tiled>::expand(buffer, builder);
                    let layout = VirtualLayout::<C, Sequence<i32>>::expand(layout, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Sequence<i32>, TensorMap<E, Tiled>>::new(
                            buffer, layout,
                        );
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapIm2col { buffer, layout } => {
                    let buffer = TensorMap::<E, Im2col>::expand(buffer, builder);
                    let layout =
                        VirtualLayout::<C, (Sequence<i32>, Sequence<i32>)>::expand(layout, builder);
                    let view = VirtualViewExpand::<
                        E,
                        C,
                        (Sequence<i32>, Sequence<i32>),
                        TensorMap<E, Im2col>,
                    >::new(buffer, layout);
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::Read(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::Quantized {
                    values,
                    scales,
                    scheme,
                } => quant::view::expand_dynamic(values, scales, *scheme, builder),
            }
        }
        fn expand_output(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            match arg {
                ViewCompilationArg::Array { buffer, layout } => {
                    let buffer = Array::<E>::expand_output(buffer, builder);
                    let layout = VirtualLayout::<C, Coords1d>::expand_output(layout, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Coords1d, Array<E>>::new(buffer, layout);
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapTiled { buffer, layout } => {
                    let buffer = TensorMap::<E, Tiled>::expand_output(buffer, builder);
                    let layout = VirtualLayout::<C, Sequence<i32>>::expand_output(layout, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Sequence<i32>, TensorMap<E, Tiled>>::new(
                            buffer, layout,
                        );
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapIm2col { .. } => {
                    unimplemented!("Im2col tensor maps can't be used as outputs");
                }
                ViewCompilationArg::Quantized { .. } => panic!("Quantized views must be readonly"),
            }
        }
    }
}

pub use dynamic::*;

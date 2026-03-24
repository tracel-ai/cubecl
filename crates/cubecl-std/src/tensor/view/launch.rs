use cubecl_core::prelude::*;
use std::{marker::PhantomData, ops::Deref, sync::Arc};

use crate::tensor::{
    View, ViewExpand, VirtualViewMutExpand,
    layout::{Coordinates, Coords1d, Layout, VirtualLayoutExpand},
    view::ViewType,
};

mod layout {
    use core::{cell::RefCell, fmt::Debug, hash::Hash};

    use alloc::rc::Rc;
    use cubecl_core::{
        self as cubecl,
        format::DebugRaw,
        hash::{StableHash, StableHasher},
        prelude::*,
        zspace::{Shape, Strides, metadata::Metadata},
    };

    use crate::tensor::layout::LayoutExpand;

    use super::*;

    #[allow(clippy::len_without_is_empty)]
    pub trait BufferArg: 'static {
        fn len(&self) -> usize;
        fn shape(&self) -> &[usize];
        fn strides(&self) -> &[usize];
    }

    impl<R: Runtime> BufferArg for TensorArg<R> {
        fn len(&self) -> usize {
            self.size()
        }

        fn shape(&self) -> &[usize] {
            self.shape()
        }

        fn strides(&self) -> &[usize] {
            self.strides()
        }
    }
    impl<R: Runtime> BufferArg for ArrayArg<R> {
        fn len(&self) -> usize {
            self.size()
        }

        fn shape(&self) -> &[usize] {
            self.shape()
        }

        fn strides(&self) -> &[usize] {
            &[1]
        }
    }
    impl<R: Runtime, K: TensorMapKind> BufferArg for TensorMapArg<R, K> {
        fn len(&self) -> usize {
            self.tensor.size()
        }

        fn shape(&self) -> &[usize] {
            self.tensor.shape()
        }

        fn strides(&self) -> &[usize] {
            self.tensor.strides()
        }
    }

    impl BufferArg for Metadata {
        fn len(&self) -> usize {
            self.shape.num_elements()
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn strides(&self) -> &[usize] {
            &self.strides
        }
    }

    /// Special launch arg that gets the handle and types of the view, to allow inferring launch
    /// state based on type/handle metadata, avoiding duplication. All `LaunchArg`s also implement
    /// this trait.
    pub trait ViewLayoutLaunchArg: CubeType + Send + Sync + 'static {
        /// The runtime argument for the kernel.
        type RuntimeArg<R: Runtime>: Send + Sync;
        /// Compilation argument.
        type CompilationArg: CompilationArg;

        fn register<R: Runtime, B: BufferArg>(
            arg: Self::RuntimeArg<R>,
            buffer: &B,
            ty: Type,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg;

        /// Register an input variable during compilation that fill the [`KernelBuilder`].
        fn expand(
            arg: &Self::CompilationArg,
            ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType;

        /// Register an output variable during compilation that fill the [`KernelBuilder`].
        fn expand_output(
            arg: &Self::CompilationArg,
            ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            Self::expand(arg, ty, builder)
        }
    }

    impl<T: LaunchArg> ViewLayoutLaunchArg for T {
        type RuntimeArg<R: Runtime> = <T as LaunchArg>::RuntimeArg<R>;
        type CompilationArg = <T as LaunchArg>::CompilationArg;

        fn register<R: Runtime, B: BufferArg>(
            arg: Self::RuntimeArg<R>,
            _buffer: &B,
            _ty: Type,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            <T as LaunchArg>::register(arg, launcher)
        }

        fn expand(
            arg: &Self::CompilationArg,
            _ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            <T as LaunchArg>::expand(arg, builder)
        }

        fn expand_output(
            arg: &Self::CompilationArg,
            _ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            <T as LaunchArg>::expand_output(arg, builder)
        }
    }

    pub struct VirtualViewLayoutLaunch<C: Coordinates, S: Coordinates, B: BufferArg, R: Runtime> {
        _ty: core::marker::PhantomData<R>,
        #[allow(clippy::type_complexity)]
        register: Box<
            dyn FnOnce(&B, Type, &mut KernelLauncher<R>) -> VirtualViewLayoutCompilationArg<C, S>
                + Send
                + Sync,
        >,
    }

    impl<C: Coordinates, S: Coordinates, B: BufferArg, R: Runtime> VirtualViewLayoutLaunch<C, S, B, R> {
        pub fn new<L: Layout<Coordinates = C, SourceCoordinates = S> + ViewLayoutLaunchArg>(
            layout: L::RuntimeArg<R>,
        ) -> Self {
            Self {
                _ty: PhantomData,
                register: Box::new(move |buffer, ty, launcher| {
                    let comp_arg = L::register::<R, B>(layout, buffer, ty, launcher);
                    let comp_arg_2 = comp_arg.clone();
                    let expand = Rc::new(RefCell::new(
                        move |ty: Type, builder: &mut KernelBuilder, is_out: bool| {
                            let expand = match is_out {
                                true => L::expand_output(&comp_arg_2, ty, builder),
                                false => L::expand(&comp_arg_2, ty, builder),
                            };
                            VirtualLayoutExpand::new(expand)
                        },
                    ));
                    VirtualViewLayoutCompilationArg::new(comp_arg, expand)
                }),
            }
        }

        pub fn register(
            self,
            buffer: &B,
            ty: Type,
            launcher: &mut KernelLauncher<R>,
        ) -> VirtualViewLayoutCompilationArg<C, S> {
            (self.register)(buffer, ty, launcher)
        }
    }

    type ExpandFn<C, S> =
        Rc<RefCell<dyn FnMut(Type, &mut KernelBuilder, bool) -> VirtualLayoutExpand<C, S> + Send>>;

    #[derive(Clone)]
    pub struct VirtualViewLayoutCompilationArg<C: Coordinates, S: Coordinates> {
        type_name: String,
        debug: Rc<dyn core::fmt::Debug>,
        hash: StableHash,
        expand: ExpandFn<C, S>,
    }

    // SAFETY: The struct is readonly, so `Sync` is safe to implement
    unsafe impl<C: Coordinates, S: Coordinates> Send for VirtualViewLayoutCompilationArg<C, S> {}
    unsafe impl<C: Coordinates, S: Coordinates> Sync for VirtualViewLayoutCompilationArg<C, S> {}

    impl<C: Coordinates, S: Coordinates> VirtualViewLayoutCompilationArg<C, S> {
        pub fn new<L: CompilationArg + 'static>(arg: L, expand: ExpandFn<C, S>) -> Self {
            // Hash ahead of time so we don't need to store the actual data, which would be far
            // more complex
            let hash = StableHasher::hash_one(&arg);
            Self {
                type_name: core::any::type_name::<L>().to_string(),
                debug: Rc::new(arg),
                hash,
                expand,
            }
        }

        pub fn expand(&self, ty: Type, builder: &mut KernelBuilder) -> VirtualLayoutExpand<C, S> {
            let mut expand = self.expand.borrow_mut();
            (expand)(ty, builder, false)
        }

        pub fn expand_output(
            &self,
            ty: Type,
            builder: &mut KernelBuilder,
        ) -> VirtualLayoutExpand<C, S> {
            let mut expand = self.expand.borrow_mut();
            (expand)(ty, builder, true)
        }
    }

    impl<C: Coordinates, S: Coordinates> PartialEq for VirtualViewLayoutCompilationArg<C, S> {
        fn eq(&self, other: &Self) -> bool {
            self.type_name == other.type_name && self.hash == other.hash
        }
    }
    impl<C: Coordinates, S: Coordinates> Eq for VirtualViewLayoutCompilationArg<C, S> {}

    impl<C: Coordinates, S: Coordinates> core::hash::Hash for VirtualViewLayoutCompilationArg<C, S> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.type_name.hash(state);
            self.hash.hash(state);
        }
    }

    impl<C: Coordinates, S: Coordinates> core::fmt::Debug for VirtualViewLayoutCompilationArg<C, S> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct(stringify!(VirtualLayout))
                .field("type", &DebugRaw(&self.type_name))
                .field("value", &self.debug)
                .finish()
        }
    }

    #[derive(CubeType)]
    pub struct ConcreteLayout<L: Layout + ViewLayoutLaunchArg> {
        value: L,
    }

    #[cube]
    impl<L: Layout + ViewLayoutLaunchArg> Layout for ConcreteLayout<L> {
        type Coordinates = L::Coordinates;
        type SourceCoordinates = L::SourceCoordinates;

        fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
            self.value.to_source_pos(pos)
        }

        fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
            self.value.to_source_pos_checked(pos)
        }

        fn shape(&self) -> Self::Coordinates {
            self.value.shape()
        }

        fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
            self.value.is_in_bounds(pos)
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> Deref for ConcreteLayout<L> {
        type Target = L;

        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> Deref for ConcreteLayoutExpand<L> {
        type Target = <L as CubeType>::ExpandType;

        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }

    pub struct ConcreteLayoutLaunch<L: Layout + ViewLayoutLaunchArg, R: Runtime> {
        meta: Metadata,
        ty: Type,
        value: L::RuntimeArg<R>,
    }

    impl<L: Layout + ViewLayoutLaunchArg, R: Runtime> ConcreteLayoutLaunch<L, R> {
        pub fn new(meta: Metadata, ty: Type, value: L::RuntimeArg<R>) -> Self {
            Self { meta, ty, value }
        }

        pub fn from_handle(handle: &TensorBinding<R>, ty: Type, value: L::RuntimeArg<R>) -> Self {
            Self {
                meta: Metadata {
                    shape: handle.shape.clone(),
                    strides: handle.strides.clone(),
                },
                ty,
                value,
            }
        }

        pub fn from_shape_strides(
            shape: Shape,
            strides: Strides,
            ty: Type,
            value: L::RuntimeArg<R>,
        ) -> Self {
            Self {
                meta: Metadata { shape, strides },
                ty,
                value,
            }
        }
    }

    pub struct ConcreteLayoutCompilationArg<L: Layout + ViewLayoutLaunchArg> {
        ty: Type,
        value: L::CompilationArg,
    }

    impl<L: Layout + ViewLayoutLaunchArg> Debug for ConcreteLayoutCompilationArg<L> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ConcreteLayoutCompilationArg")
                .field("ty", &self.ty)
                .field("value", &self.value)
                .finish()
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> Hash for ConcreteLayoutCompilationArg<L> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.ty.hash(state);
            self.value.hash(state);
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> Eq for ConcreteLayoutCompilationArg<L> {}
    impl<L: Layout + ViewLayoutLaunchArg> PartialEq for ConcreteLayoutCompilationArg<L> {
        fn eq(&self, other: &Self) -> bool {
            self.ty == other.ty && self.value == other.value
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> Clone for ConcreteLayoutCompilationArg<L> {
        fn clone(&self) -> Self {
            Self {
                ty: self.ty,
                value: self.value.clone(),
            }
        }
    }

    impl<L: Layout + ViewLayoutLaunchArg> LaunchArg for ConcreteLayout<L> {
        type RuntimeArg<R: Runtime> = ConcreteLayoutLaunch<L, R>;
        type CompilationArg = ConcreteLayoutCompilationArg<L>;

        fn register<R: Runtime>(
            arg: Self::RuntimeArg<R>,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            ConcreteLayoutCompilationArg {
                value: L::register(arg.value, &arg.meta, arg.ty, launcher),
                ty: arg.ty,
            }
        }

        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            ConcreteLayoutExpand {
                value: L::expand(&arg.value, arg.ty, builder),
            }
        }

        fn expand_output(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            ConcreteLayoutExpand {
                value: L::expand_output(&arg.value, arg.ty, builder),
            }
        }
    }
}

pub use layout::*;

mod dynamic {
    use cubecl_common::quant::scheme::QuantScheme;

    use crate::{
        quant::{
            self,
            view::{RegisterDynamic, run_with_quant_type},
        },
        tensor::{
            VirtualViewExpand,
            launch::layout::{ViewLayoutLaunchArg, VirtualViewLayoutLaunch},
            layout::as_dyn::{IntoDyn, IntoDyn2Layout, IntoDynLayout},
        },
    };

    use super::*;

    #[allow(clippy::type_complexity)]
    pub enum ViewArg<C: Coordinates, R: Runtime> {
        Array(
            ArrayArg<R>,
            VirtualViewLayoutLaunch<C, Coords1d, ArrayArg<R>, R>,
        ),
        Tensor(
            TensorArg<R>,
            VirtualViewLayoutLaunch<C, Coords1d, TensorArg<R>, R>,
        ),
        TensorMapTiled(
            TensorMapArg<R, Tiled>,
            VirtualViewLayoutLaunch<C, Sequence<i32>, TensorMapArg<R, Tiled>, R>,
        ),
        TensorMapIm2col(
            TensorMapArg<R, Im2col>,
            VirtualViewLayoutLaunch<C, (Sequence<i32>, Sequence<i32>), TensorMapArg<R, Im2col>, R>,
        ),
        Quantized {
            values: Box<ViewArg<C, R>>,
            scales: Box<ViewArg<C, R>>,
            scheme: QuantScheme,
        },
    }

    impl<C: Coordinates, R: Runtime> ViewArg<C, R> {
        pub fn new_array<
            L: Layout<Coordinates = C, SourceCoordinates = Coords1d> + ViewLayoutLaunchArg,
        >(
            buffer: ArrayArg<R>,
            layout: L::RuntimeArg<R>,
        ) -> Self {
            let layout = VirtualViewLayoutLaunch::new::<L>(layout);
            ViewArg::Array(buffer, layout)
        }

        pub fn new_tensor<
            L: Layout<Coordinates = C, SourceCoordinates = Coords1d> + ViewLayoutLaunchArg,
        >(
            buffer: TensorArg<R>,
            layout: L::RuntimeArg<R>,
        ) -> Self {
            let layout = VirtualViewLayoutLaunch::new::<L>(layout);
            ViewArg::Tensor(buffer, layout)
        }

        pub fn new_tensor_map_tiled<
            L: Layout<Coordinates = C, SourceCoordinates: IntoDyn> + ViewLayoutLaunchArg,
        >(
            buffer: TensorMapArg<R, Tiled>,
            layout: L::RuntimeArg<R>,
        ) -> ViewArg<C, R> {
            let layout = VirtualViewLayoutLaunch::new::<IntoDynLayout<L>>(layout);
            ViewArg::TensorMapTiled(buffer, layout)
        }

        pub fn new_tensor_map_im2col<
            L: Layout<Coordinates = C, SourceCoordinates = (P, O)> + ViewLayoutLaunchArg,
            P: IntoDyn,
            O: IntoDyn,
        >(
            buffer: TensorMapArg<R, Im2col>,
            layout: L::RuntimeArg<R>,
        ) -> ViewArg<C, R> {
            let layout = VirtualViewLayoutLaunch::new::<IntoDyn2Layout<L, P, O>>(layout);
            ViewArg::TensorMapIm2col(buffer, layout)
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
    #[derive(Clone)]
    pub enum ViewCompilationArg<C: Coordinates> {
        Array {
            buffer: ArrayCompilationArg,
            layout: VirtualViewLayoutCompilationArg<C, Coords1d>,
        },
        TensorMapTiled {
            buffer: (),
            layout: VirtualViewLayoutCompilationArg<C, Sequence<i32>>,
        },
        TensorMapIm2col {
            buffer: (),
            layout: VirtualViewLayoutCompilationArg<C, (Sequence<i32>, Sequence<i32>)>,
        },
        Quantized {
            values: Box<ViewCompilationArg<C>>,
            scales: Box<ViewCompilationArg<C>>,
            scheme: QuantScheme,
        },
    }

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
        type RuntimeArg<R: Runtime> = ViewArg<C, R>;
        type CompilationArg = ViewCompilationArg<C>;

        fn register<R: Runtime>(
            arg: Self::RuntimeArg<R>,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            let ty = launcher.with_scope(|scope| E::as_type(scope));
            match arg {
                ViewArg::Array(buffer, layout) => ViewCompilationArg::Array {
                    layout: layout.register(&buffer, ty, launcher),
                    buffer: <Array<E> as LaunchArg>::register(buffer, launcher),
                },
                ViewArg::Tensor(buffer, layout) => ViewCompilationArg::Array {
                    layout: layout.register(&buffer, ty, launcher),
                    buffer: <Array<E> as LaunchArg>::register(buffer.into_array_arg(), launcher),
                },
                ViewArg::TensorMapTiled(buffer, layout) => ViewCompilationArg::TensorMapTiled {
                    layout: layout.register(&buffer, ty, launcher),
                    buffer: <TensorMap<E, Tiled> as LaunchArg>::register(buffer, launcher),
                },
                ViewArg::TensorMapIm2col(buffer, layout) => ViewCompilationArg::TensorMapIm2col {
                    layout: layout.register(&buffer, ty, launcher),
                    buffer: <TensorMap<E, Im2col> as LaunchArg>::register(buffer, launcher),
                },
                ViewArg::Quantized {
                    values,
                    scales,
                    scheme,
                } => {
                    let register = RegisterDynamic {
                        values: *values,
                        scales: *scales,
                        scheme,
                        launcher,
                        _ty: PhantomData::<E>,
                    };
                    run_with_quant_type(register, scheme)
                }
            }
        }
        fn expand(
            arg: &Self::CompilationArg,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            let ty = E::as_type(&builder.scope);
            match arg {
                ViewCompilationArg::Array { buffer, layout } => {
                    let layout = layout.expand(ty, builder);
                    let buffer = <Array<E> as LaunchArg>::expand(buffer, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Coords1d, Array<E>>::new(buffer, layout);
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapTiled { buffer, layout } => {
                    let layout = layout.expand(ty, builder);
                    let buffer = <TensorMap<E, Tiled> as LaunchArg>::expand(buffer, builder);
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
                    let layout = layout.expand(ty, builder);
                    let buffer = <TensorMap<E, Im2col> as LaunchArg>::expand(buffer, builder);
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
            let ty = E::as_type(&builder.scope);
            match arg {
                ViewCompilationArg::Array { buffer, layout } => {
                    let layout = layout.expand_output(ty, builder);
                    let buffer = <Array<E> as LaunchArg>::expand_output(buffer, builder);
                    let view =
                        VirtualViewMutExpand::<E, C, Coords1d, Array<E>>::new(buffer, layout);
                    ViewExpand::<E, C, IO> {
                        inner: ViewType::ReadWrite(Arc::new(view)),
                        _io: PhantomData,
                    }
                }
                ViewCompilationArg::TensorMapTiled { buffer, layout } => {
                    let layout = layout.expand_output(ty, builder);
                    let buffer = <TensorMap<E, Tiled> as LaunchArg>::expand_output(buffer, builder);
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

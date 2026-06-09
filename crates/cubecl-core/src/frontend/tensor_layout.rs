use alloc::boxed::Box;

use cubecl_ir::{ClampMode, Instruction, SemanticType, TensorIndexingOps};

use crate::{self as cubecl, unexpanded};

use crate::prelude::*;

#[derive_cube_comptime]
pub enum TensorClampMode {
    Undefined,
    Constant(u32),
    ClampToEdge,
    Repeat,
    RepeatMirrored,
}

impl From<TensorClampMode> for ClampMode {
    fn from(value: TensorClampMode) -> Self {
        match value {
            TensorClampMode::Undefined => ClampMode::Undefined,
            TensorClampMode::Constant(val) => ClampMode::Constant(val),
            TensorClampMode::ClampToEdge => ClampMode::ClampToEdge,
            TensorClampMode::Repeat => ClampMode::Repeat,
            TensorClampMode::RepeatMirrored => ClampMode::RepeatMirrored,
        }
    }
}

// OpTypeTensorLayoutNV with optional OpTypeTensorViewNV
#[derive(CubeType, Clone)]
pub struct TensorView<T: CubePrimitive> {
    #[allow(unused)]
    pub(crate) buffer: Box<[T]>,
    #[allow(unused)]
    pub(crate) layout: TensorLayout,
    #[allow(unused)]
    pub(crate) view: ComptimeOption<TensorReinterpret>,
}

#[derive_cube_comptime]
pub struct TensorLayout;

#[derive_cube_comptime]
pub struct TensorReinterpret;

impl CubeType for TensorLayout {
    type ExpandType = NativeExpand<TensorLayout>;
}

impl CubeDebug for TensorLayout {}
impl CubePrimitive for TensorLayout {
    type Scalar = u32;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn from_const_value(_: cubecl_ir::ConstantValue) -> Self {
        panic!("Can't construct tensor layout from constant")
    }
}

impl NativeAssign for TensorLayout {}

impl CubeType for TensorReinterpret {
    type ExpandType = NativeExpand<TensorReinterpret>;
}

impl CubeDebug for TensorReinterpret {}
impl CubePrimitive for TensorReinterpret {
    type Scalar = u32;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn from_const_value(_: cubecl_ir::ConstantValue) -> Self {
        panic!("Can't construct tensor layout from constant")
    }
}

impl NativeAssign for TensorReinterpret {}

#[derive(CubeType, CubeLaunch)]
pub struct TensorViewBuilder<T: CubePrimitive> {
    #[allow(unused)]
    buffer: Box<[T]>,
    #[allow(unused)]
    shape: Sequence<u32>,
    /// Strides default to contiguous strides
    strides: ComptimeOption<Sequence<u32>>,
    #[cube(comptime)]
    clamp_mode: TensorClampMode,
}

#[cube]
impl<T: CubePrimitive> TensorView<T> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(buffer: &[T], shape: Sequence<u32>) -> TensorViewBuilder<T> {
        TensorViewBuilder::<T> {
            buffer: unsafe { buffer.as_boxed_unchecked() },
            shape,
            strides: ComptimeOption::new_None(),
            clamp_mode: comptime![TensorClampMode::Constant(0)],
        }
    }

    #[allow(unused)]
    pub fn slice(&self, offsets: Sequence<u32>, shape: Sequence<u32>) -> TensorView<T> {
        intrinsic!(|scope| {
            assert_eq!(
                offsets.len(),
                self.layout.rank(),
                "Offsets and view rank must match"
            );
            assert_eq!(
                offsets.len(),
                shape.len(),
                "Offsets and shape must have same rank"
            );
            let new_layout = scope.create_local(self.layout.expand.ty);
            scope.register(Instruction::new(
                TensorIndexingOps::Slice {
                    layout: self.layout.expand,
                    offsets: offsets.iter_cloned().map(|it| it.expand).collect(),
                    shape: shape.iter_cloned().map(|it| it.expand).collect(),
                },
                new_layout,
            ));
            TensorViewExpand {
                buffer: self.buffer.clone(),
                layout: new_layout.into(),
                view: self.view.clone(),
            }
        })
    }
}

impl NativeExpand<TensorLayout> {
    fn rank(&self) -> usize {
        if let Type::Semantic(SemanticType::TensorLayout(rank, _)) = &self.expand.ty {
            *rank
        } else {
            unreachable!()
        }
    }
}

impl<T: CubePrimitive> TensorView<T> {
    pub fn permuted(&self, _permutation: Sequence<usize>) -> TensorView<T> {
        unexpanded!()
    }
}

impl<T: CubePrimitive> TensorViewExpand<T> {
    pub fn __expand_permuted_method(
        self,
        scope: &Scope,
        permutation: SequenceExpand<usize>,
    ) -> TensorViewExpand<T> {
        let dims = permutation.len();
        assert!(dims <= 5, "Max 5 dims allowed");
        let permutation = permutation
            .iter_cloned()
            .map(|it| {
                it.constant()
                    .expect("permutation must be constant")
                    .as_u32()
            })
            .collect::<alloc::vec::Vec<_>>();
        let mut perm_dims = [0; 5];
        perm_dims[..dims].copy_from_slice(&permutation);
        let view = scope.create_local(Type::semantic(SemanticType::TensorView(
            dims, false, perm_dims,
        )));
        scope.register(Instruction::new(TensorIndexingOps::CreateView, view));
        TensorViewExpand {
            buffer: self.buffer,
            layout: self.layout,
            view: ComptimeOptionExpand::Some(view.into()),
        }
    }
}

impl<T: CubePrimitive> TensorViewBuilder<T> {
    pub fn with_strides(mut self, strides: Sequence<u32>) -> Self {
        self.strides = ComptimeOption::Some(strides);
        self
    }

    pub fn with_clamp_mode(mut self, clamp_mode: TensorClampMode) -> Self {
        self.clamp_mode = clamp_mode;
        self
    }

    pub fn finish(self) -> TensorView<T> {
        unexpanded!()
    }
}

impl<T: CubePrimitive> TensorViewBuilderExpand<T> {
    pub fn __expand_with_strides_method(
        mut self,
        _scope: &Scope,
        strides: SequenceExpand<u32>,
    ) -> Self {
        self.strides = ComptimeOptionExpand::Some(strides);
        self
    }

    pub fn __expand_with_clamp_mode_method(
        mut self,
        _scope: &Scope,
        clamp_mode: TensorClampMode,
    ) -> Self {
        self.clamp_mode = clamp_mode;
        self
    }

    pub fn __expand_finish_method(self, scope: &Scope) -> TensorViewExpand<T> {
        let layout = scope.create_local(Type::semantic(SemanticType::TensorLayout(
            self.shape.len(),
            self.clamp_mode.into(),
        )));
        scope.register(Instruction::new(
            TensorIndexingOps::CreateLayout {
                shape: self.shape.iter_cloned().map(|it| it.expand).collect(),
                strides: match self.strides {
                    ComptimeOptionExpand::None => None,
                    ComptimeOptionExpand::Some(strides) => {
                        Some(strides.iter_cloned().map(|it| it.expand).collect())
                    }
                },
                clamp_mode: self.clamp_mode.into(),
            },
            layout,
        ));
        TensorViewExpand {
            buffer: self.buffer,
            layout: layout.into(),
            view: ComptimeOptionExpand::None,
        }
    }
}

impl<T: CubePrimitive> LaunchArg for TensorView<T> {
    type RuntimeArg<R: Runtime> = TensorViewBuilderLaunch<T, R>;
    type CompilationArg = TensorViewBuilderCompilationArg<T>;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        TensorViewBuilder::<T>::register(arg, launcher)
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let build = TensorViewBuilder::<T>::expand(arg, builder);
        build.__expand_finish_method(&builder.scope)
    }
}

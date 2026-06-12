use alloc::boxed::Box;

use cubecl_ir::{
    dialect::spirv::{CreateLayoutOp, CreateViewOp, SliceOp},
    pliron::{
        builtin::op_interfaces::OneResultInterface,
        context::Ptr,
        r#type::{TypeObj, Typed},
    },
    types::spirv::{ClampMode, TensorLayoutType, TensorViewType},
};

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

    fn __expand_as_type(_scope: &Scope) -> Ptr<TypeObj> {
        unimplemented!()
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

    fn __expand_as_type(_scope: &Scope) -> Ptr<TypeObj> {
        unimplemented!()
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
    pub fn slice(&self, offs: Sequence<u32>, shape: Sequence<u32>) -> TensorView<T> {
        intrinsic!(|scope| {
            assert_eq!(
                offs.len(),
                self.layout.rank(scope),
                "Offsets and view rank must match"
            );
            assert_eq!(
                offs.len(),
                shape.len(),
                "Offsets and shape must have same rank"
            );
            let layout = self.layout.value(scope);
            let offs = offs.iter_cloned().map(|it| it.read_value(scope)).collect();
            let shape = shape.iter_cloned().map(|it| it.read_value(scope)).collect();
            let slice_op = SliceOp::new(&mut scope.ctx_mut(), layout, offs, shape);
            scope.register(&slice_op);
            let new_layout = slice_op.get_result(&scope.ctx());
            TensorViewExpand {
                buffer: self.buffer.clone(),
                layout: new_layout.into(),
                view: self.view.clone(),
            }
        })
    }
}

impl NativeExpand<TensorLayout> {
    fn rank(&self, scope: &Scope) -> usize {
        let ty = self.read_value(scope).get_type(&scope.ctx());
        let ctx = scope.ctx();
        let ty = ty.deref(&ctx);
        let TensorLayoutType { rank, .. } = ty.downcast_ref().unwrap();
        *rank
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
                    .as_usize()
            })
            .collect::<alloc::vec::Vec<_>>();
        let ty = TensorViewType::get(&mut scope.ctx_mut(), permutation.len(), false, permutation);
        let op = CreateViewOp::new(&mut scope.ctx_mut(), ty.into());
        scope.register(&op);
        let view = op.get_result(&scope.ctx());

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
        let shape = self.shape.into_iter().map(|it| it.read_value(scope));
        let strides = match self.strides {
            ComptimeOptionExpand::None => None,
            ComptimeOptionExpand::Some(strides) => {
                Some(strides.into_iter().map(|it| it.read_value(scope)).collect())
            }
        };
        let clamp_mode = ClampMode::from(self.clamp_mode);

        let op = CreateLayoutOp::new(&mut scope.ctx_mut(), shape.collect(), strides, clamp_mode);
        scope.register(&op);
        let layout = op.get_result(&scope.ctx());

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

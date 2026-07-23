use cubecl::prelude::*;
use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{
    dialect::general::ReadScalarOp,
    pliron::{builtin::op_interfaces::OneResultInterface, value::Value},
};
use serde::{Deserialize, Serialize};

use crate::{
    self as cubecl, ScalarArgType, intrinsic,
    ir::{ElemType, FloatKind, IntKind, UIntKind},
};

#[derive(Clone, Copy, Debug)]
/// A way to define an input scalar without a generic attached to it.
///
/// It uses comptime enum with zero-cost runtime abstraction for kernel generation.
pub struct InputScalar {
    data: [u8; 8],
    dtype: ElemType,
}

#[derive(Clone)]
pub struct InputScalarExpand {
    pub expand: Value,
}

impl CubeType for InputScalar {
    type ExpandType = InputScalarExpand;
}

impl ExpandTypeClone for InputScalarExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl IntoExpand for InputScalarExpand {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}

impl IntoMut for InputScalarExpand {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl CubeDebug for InputScalarExpand {}

impl AsRefExpand for InputScalarExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for InputScalarExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl InputScalar {
    /// Creates an [`InputScalar`] from the given element and dtype.
    ///
    /// # Panics
    ///
    /// If the given numeric element can't be transformed into the passed [`ElemType`].
    pub fn new<E: num_traits::ToPrimitive>(val: E, dtype: impl Into<ElemType>) -> Self {
        let dtype: ElemType = dtype.into();
        let mut out = InputScalar {
            data: Default::default(),
            dtype,
        };
        fn write<E: ScalarArgType>(val: impl num_traits::ToPrimitive, out: &mut [u8]) {
            let val = [E::from(val).unwrap()];
            let bytes = E::as_bytes(&val);
            out[..bytes.len()].copy_from_slice(bytes);
        }
        match dtype {
            ElemType::Index => panic!(
                "Index is not supported as a scalar storage type. Use the address type's `unsigned_type()` instead."
            ),
            ElemType::Float(float_kind) => match float_kind {
                FloatKind::F16 => write::<half::f16>(val, &mut out.data),
                FloatKind::BF16 => write::<half::bf16>(val, &mut out.data),
                FloatKind::Flex32 | FloatKind::F32 | FloatKind::TF32 => {
                    write::<f32>(val, &mut out.data)
                }
                FloatKind::F64 => write::<f64>(val, &mut out.data),
                FloatKind::E2M1 | FloatKind::E2M1x2 | FloatKind::E2M3 | FloatKind::E3M2 => {
                    unimplemented!("fp6 CPU conversion not yet implemented")
                }
                FloatKind::E4M3 => write::<e4m3>(val, &mut out.data),
                FloatKind::E5M2 => write::<e5m2>(val, &mut out.data),
                FloatKind::UE8M0 => write::<ue8m0>(val, &mut out.data),
            },
            ElemType::Int(int_kind) => match int_kind {
                IntKind::I8 => write::<i8>(val, &mut out.data),
                IntKind::I16 => write::<i16>(val, &mut out.data),
                IntKind::I32 => write::<i32>(val, &mut out.data),
                IntKind::I64 => write::<i64>(val, &mut out.data),
            },
            ElemType::UInt(uint_kind) => match uint_kind {
                UIntKind::U8 => write::<u8>(val, &mut out.data),
                UIntKind::U16 => write::<u16>(val, &mut out.data),
                UIntKind::U32 => write::<u32>(val, &mut out.data),
                UIntKind::U64 => write::<u64>(val, &mut out.data),
            },
            ElemType::Bool => panic!("Bool isn't a scalar"),
        };
        out
    }
}

#[cube]
impl InputScalar {
    /// Reads the scalar with the given element type.
    ///
    /// Performs casting if necessary.
    pub fn get<C: Scalar>(&self) -> C {
        intrinsic!(|scope| {
            let dtype = C::__expand_as_type(scope);
            cast_value(scope, self.expand, dtype).into()
        })
    }
}

impl InputScalar {
    pub fn as_bytes(&self) -> &[u8] {
        // Address type is irrelevant since we don't allow it as a dtype
        &self.data[..self.dtype.size()]
    }
}

impl LaunchArg for InputScalar {
    type RuntimeArg<R: Runtime> = InputScalar;
    type CompilationArg = InputScalarCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let dtype = arg.dtype;

        launcher.register_scalar_raw(arg.as_bytes(), dtype);
        InputScalarCompilationArg::new(arg.dtype)
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let id = builder.scalar(arg.ty);
        let ty = arg.ty.to_type(builder.ctx_mut());
        let op = ReadScalarOp::new(builder.ctx_mut(), ty, id);
        builder.register(&op);
        let expand = op.get_result(builder.ctx());
        InputScalarExpand { expand }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct InputScalarCompilationArg {
    ty: ElemType,
}

impl InputScalarCompilationArg {
    pub fn new(ty: ElemType) -> Self {
        Self { ty }
    }
}

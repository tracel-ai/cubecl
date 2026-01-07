use cubecl::prelude::*;
use cubecl_common::{e4m3, e5m2, ue8m0};
use serde::{Deserialize, Serialize};

use crate::{
    self as cubecl, CubeScalar, intrinsic,
    ir::{ElemType, ExpandElement, FloatKind, IntKind, Type, UIntKind},
};

#[derive(Clone, Copy)]
/// A way to define an input scalar without a generic attached to it.
///
/// It uses comptime enum with zero-cost runtime abstraction for kernel generation.
pub struct InputScalar {
    data: [u8; 8],
    dtype: StorageType,
}

#[derive(Clone)]
pub struct InputScalarExpand {
    pub expand: ExpandElement,
}

impl CubeType for InputScalar {
    type ExpandType = InputScalarExpand;
}

impl IntoMut for InputScalarExpand {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for InputScalarExpand {}

impl InputScalar {
    /// Creates an [InputScalar] from the given element and dtype.
    ///
    /// # Panics
    ///
    /// If the given numeric element can't be transformed into the passed [ElemType].
    pub fn new<E: num_traits::ToPrimitive>(val: E, dtype: impl Into<StorageType>) -> Self {
        let dtype: StorageType = dtype.into();
        let mut out = InputScalar {
            data: Default::default(),
            dtype,
        };
        fn write<E: CubeScalar>(val: impl num_traits::ToPrimitive, out: &mut [u8]) {
            let val = [E::from(val).unwrap()];
            let bytes = E::as_bytes(&val);
            out[..bytes.len()].copy_from_slice(bytes);
        }
        match dtype {
            StorageType::Scalar(elem) => match elem {
                ElemType::Float(float_kind) => match float_kind {
                    FloatKind::F16 => write::<half::f16>(val, &mut out.data),
                    FloatKind::BF16 => write::<half::bf16>(val, &mut out.data),
                    FloatKind::Flex32 | FloatKind::F32 | FloatKind::TF32 => {
                        write::<f32>(val, &mut out.data)
                    }
                    FloatKind::F64 => write::<f64>(val, &mut out.data),
                    FloatKind::E2M1 | FloatKind::E2M3 | FloatKind::E3M2 => {
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
            },
            other => unimplemented!("{other} not supported for scalars"),
        };
        out
    }
}

#[cube]
impl InputScalar {
    /// Reads the scalar with the given element type.
    ///
    /// Performs casting if necessary.
    pub fn get<C: CubePrimitive>(&self) -> C {
        intrinsic!(|scope| {
            let dtype = C::as_type(scope);
            if self.expand.storage_type() == dtype {
                return self.expand.into();
            }
            let new_var = scope.create_local(Type::new(dtype));
            cast::expand::<C, C>(scope, self.expand.into(), new_var.clone().into());
            new_var.into()
        })
    }
}

impl InputScalar {
    pub fn as_bytes(&self) -> Vec<u8> {
        self.data[..self.dtype.size()].to_vec()
    }
}

impl LaunchArg for InputScalar {
    type RuntimeArg<'a, R: Runtime> = InputScalar;
    type CompilationArg = InputScalarCompilationArg;

    fn compilation_arg<R: Runtime>(arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        InputScalarCompilationArg::new(arg.dtype)
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let expand = builder.scalar(arg.ty);
        InputScalarExpand { expand }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct InputScalarCompilationArg {
    ty: StorageType,
}

impl InputScalarCompilationArg {
    pub fn new(ty: StorageType) -> Self {
        Self { ty }
    }
}

impl CompilationArg for InputScalarCompilationArg {}

impl<R: Runtime> ArgSettings<R> for InputScalar {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        let dtype = self.dtype;
        launcher.register_scalar_raw(&self.data[..dtype.size()], dtype);
    }
}

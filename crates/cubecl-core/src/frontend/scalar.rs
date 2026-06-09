use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::Variable;
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
    dtype: StorageType,
}

#[derive(Clone)]
pub struct InputScalarExpand {
    pub expand: Variable,
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
    pub fn new<E: num_traits::ToPrimitive>(val: E, dtype: impl Into<StorageType>) -> Self {
        let dtype: StorageType = dtype.into();
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
    pub fn get<C: Scalar>(&self) -> C {
        intrinsic!(|scope| {
            let dtype = C::__expand_as_type(scope);
            if self.expand.ty == dtype {
                return self.expand.clone().into();
            }
            let new_var = scope.create_local(dtype);
            cast::expand::<C, C>(scope, self.expand.clone().into(), new_var.clone().into());
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
    type RuntimeArg<R: Runtime> = InputScalar;
    type CompilationArg = InputScalarCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let dtype = arg.dtype;
        launcher.register_scalar_raw(&arg.data[..dtype.size()], dtype);
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

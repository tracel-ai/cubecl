use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, intrinsic,
    ir::{ElemType, ExpandElement, FloatKind, IntKind, UIntKind},
};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(CubeType, Clone)]
/// A way to define an input scalar without a generic attached to it.
///
/// It uses comptime enum with zero-cost runtime abstraction for kernel generation.
pub enum InputScalar {
    F64(f64),
    F32(f32),
    F16(f16),
    BF16(bf16),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

impl InputScalar {
    /// Creates an [InputScalar] from the given element and dtype.
    ///
    /// # Panics
    ///
    /// If the given numeric element can't be transformed into the passed [ElemType].
    pub fn new<E: Numeric>(val: E, dtype: impl Into<ElemType>) -> Self {
        let dtype: ElemType = dtype.into();
        match dtype {
            ElemType::Float(float_kind) => match float_kind {
                FloatKind::F16 => Self::F16(half::f16::from_f32(val.to_f32().unwrap())),
                FloatKind::BF16 => Self::BF16(half::bf16::from_f32(val.to_f32().unwrap())),
                FloatKind::Flex32 | FloatKind::F32 | FloatKind::TF32 => {
                    Self::F32(val.to_f32().unwrap())
                }
                FloatKind::F64 => Self::F64(val.to_f64().unwrap()),
                _ => panic!("Unsupported float element type"),
            },
            ElemType::Int(int_kind) => match int_kind {
                IntKind::I8 => Self::I8(val.to_i8().unwrap()),
                IntKind::I16 => Self::I16(val.to_i16().unwrap()),
                IntKind::I32 => Self::I32(val.to_i32().unwrap()),
                IntKind::I64 => Self::I64(val.to_i64().unwrap()),
            },
            ElemType::UInt(uint_kind) => match uint_kind {
                UIntKind::U8 => Self::U8(val.to_u8().unwrap()),
                UIntKind::U16 => Self::U16(val.to_u16().unwrap()),
                UIntKind::U32 => Self::U32(val.to_u32().unwrap()),
                UIntKind::U64 => Self::U64(val.to_u64().unwrap()),
            },
            ElemType::Bool => panic!("Bool isn't a scalar"),
        }
    }
}

#[cube]
impl InputScalar {
    /// Reads the scalar with the given element type.
    ///
    /// Performs casting if necessary.
    pub fn get<C: CubePrimitive>(&self) -> C {
        intrinsic!(|scope| {
            let dtype = C::as_type(scope).elem_type();

            match self {
                InputScalarExpand::U64(val) => {
                    if dtype == ElemType::UInt(cubecl::ir::UIntKind::U64) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::U32(val) => {
                    if dtype == ElemType::UInt(cubecl::ir::UIntKind::U32) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::U16(val) => {
                    if dtype == ElemType::UInt(cubecl::ir::UIntKind::U16) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::F64(val) => {
                    if dtype == ElemType::Float(cubecl::ir::FloatKind::F64) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::F32(val) => {
                    if dtype == ElemType::Float(cubecl::ir::FloatKind::F32) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::F16(val) => {
                    if dtype == ElemType::Float(cubecl::ir::FloatKind::F16) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::BF16(val) => {
                    if dtype == ElemType::Float(cubecl::ir::FloatKind::BF16) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::U8(val) => {
                    if dtype == ElemType::UInt(cubecl::ir::UIntKind::U8) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }

                InputScalarExpand::I64(val) => {
                    if dtype == ElemType::Int(cubecl::ir::IntKind::I64) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::I32(val) => {
                    if dtype == ElemType::Int(cubecl::ir::IntKind::I32) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::I16(val) => {
                    if dtype == ElemType::Int(cubecl::ir::IntKind::I16) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
                InputScalarExpand::I8(val) => {
                    if dtype == ElemType::Int(cubecl::ir::IntKind::I8) {
                        let expand: ExpandElement = val.clone().into();
                        ExpandElementTyped::from(expand.clone())
                    } else {
                        C::__expand_cast_from(scope, val.clone())
                    }
                }
            }
        })
    }
}

impl LaunchArg for InputScalar {
    type RuntimeArg<'a, R: Runtime> = InputScalar;
    type CompilationArg = InputScalarCompilationArg;

    fn compilation_arg<R: Runtime>(arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match arg {
            InputScalar::F64(_) => {
                InputScalarCompilationArg::new(ElemType::Float(FloatKind::F64).into())
            }
            InputScalar::F32(_) => {
                InputScalarCompilationArg::new(ElemType::Float(FloatKind::F32).into())
            }
            InputScalar::F16(_) => {
                InputScalarCompilationArg::new(ElemType::Float(FloatKind::F16).into())
            }
            InputScalar::BF16(_) => {
                InputScalarCompilationArg::new(ElemType::Float(FloatKind::BF16).into())
            }
            InputScalar::I64(_) => {
                InputScalarCompilationArg::new(ElemType::Int(IntKind::I64).into())
            }
            InputScalar::I32(_) => {
                InputScalarCompilationArg::new(ElemType::Int(IntKind::I32).into())
            }
            InputScalar::I16(_) => {
                InputScalarCompilationArg::new(ElemType::Int(IntKind::I16).into())
            }
            InputScalar::I8(_) => InputScalarCompilationArg::new(ElemType::Int(IntKind::I8).into()),
            InputScalar::U64(_) => {
                InputScalarCompilationArg::new(ElemType::UInt(UIntKind::U64).into())
            }
            InputScalar::U32(_) => {
                InputScalarCompilationArg::new(ElemType::UInt(UIntKind::U32).into())
            }
            InputScalar::U16(_) => {
                InputScalarCompilationArg::new(ElemType::UInt(UIntKind::U16).into())
            }
            InputScalar::U8(_) => {
                InputScalarCompilationArg::new(ElemType::UInt(UIntKind::U8).into())
            }
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let expand = builder.scalar(arg.ty);
        match arg.ty.elem_type() {
            ElemType::Float(float_kind) => match float_kind {
                FloatKind::F16 => InputScalarExpand::F16(expand.into()),
                FloatKind::BF16 => InputScalarExpand::BF16(expand.into()),
                FloatKind::Flex32 => InputScalarExpand::F32(expand.into()),
                FloatKind::F32 => InputScalarExpand::F32(expand.into()),
                FloatKind::TF32 => InputScalarExpand::F32(expand.into()),
                FloatKind::F64 => InputScalarExpand::F32(expand.into()),
                FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0 => unimplemented!("FP8 can't be passed as scalar"),
            },
            ElemType::Int(int_kind) => match int_kind {
                IntKind::I8 => InputScalarExpand::I8(expand.into()),
                IntKind::I16 => InputScalarExpand::I16(expand.into()),
                IntKind::I32 => InputScalarExpand::I32(expand.into()),
                IntKind::I64 => InputScalarExpand::I64(expand.into()),
            },
            ElemType::UInt(uint_kind) => match uint_kind {
                UIntKind::U8 => InputScalarExpand::U8(expand.into()),
                UIntKind::U16 => InputScalarExpand::U16(expand.into()),
                UIntKind::U32 => InputScalarExpand::U32(expand.into()),
                UIntKind::U64 => InputScalarExpand::U64(expand.into()),
            },
            ElemType::Bool => panic!("Bool should be converted first."),
        }
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
        match self {
            InputScalar::F64(val) => launcher.register_f64(*val),
            InputScalar::F32(val) => launcher.register_f32(*val),
            InputScalar::F16(val) => launcher.register_f16(*val),
            InputScalar::BF16(val) => launcher.register_bf16(*val),
            InputScalar::I64(val) => launcher.register_i64(*val),
            InputScalar::I32(val) => launcher.register_i32(*val),
            InputScalar::I16(val) => launcher.register_i16(*val),
            InputScalar::I8(val) => launcher.register_i8(*val),
            InputScalar::U64(val) => launcher.register_u64(*val),
            InputScalar::U32(val) => launcher.register_u32(*val),
            InputScalar::U16(val) => launcher.register_u16(*val),
            InputScalar::U8(val) => launcher.register_u8(*val),
        }
    }
}

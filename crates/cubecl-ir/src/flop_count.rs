use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use paste::paste;

use crate::{
    ArithmeticOpCode, AtomicBinaryOperands, AtomicOp, ElemType, FloatKind, IndexOperands,
    Instruction, IntKind, Memory, Operation, OperationReflect, Processor, Scope, ScopeProcessing,
    Type, UIntKind, Value,
};

macro_rules! arith_ops_count {
    ($($op:ident),*) => {
        paste! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct ArithOpsCount {
                $(pub [< $op:lower >]: u32),*
            }

            impl ArithOpsCount {
                pub const LEN: usize = size_of::<Self>() / size_of::<u32>();

                pub fn offset(op_code: &ArithmeticOpCode) -> Option<usize> {
                    enum OpIndex {
                        $($op),*
                    }

                    Some(match op_code {
                        $(ArithmeticOpCode::$op => OpIndex::$op as usize),*
                    })
                }

                /// Whether every counter in this block is zero.
                pub fn is_empty(&self) -> bool {
                    true $(&& self.[< $op:lower >] == 0)*
                }
            }

            impl core::fmt::Display for ArithOpsCount {
                /// Prints each non-zero op as `Name: count`, comma separated. Zero counts are
                /// omitted, so an all-zero block prints nothing.
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let mut first = true;
                    $(
                        if self.[< $op:lower >] != 0 {
                            if !first {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}: {}", stringify!($op), self.[< $op:lower >])?;
                            first = false;
                        }
                    )*
                    Ok(())
                }
            }
        }
    }
}

macro_rules! ops_counts {
    (
       ariths  : [$($arith:ident),*],
       elems: [$($elems:ty),*]
    ) => {
    paste! {
            arith_ops_count!($($arith),*);

            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct OpsCount {
                pub arith: ArithOpsCount,
            }

            impl OpsCount {
                pub const LEN: usize = size_of::<Self>() / size_of::<u32>();

                pub fn offset(op: &Operation) -> Option<usize> {
                    match op {
                        Operation::Arithmetic(arith) => {
                            let base = core::mem::offset_of!(OpsCount, arith) / size_of::<u32>();
                            ArithOpsCount::offset(&arith.op_code()).map(|o| base + o)
                        }
                        _ => None,
                    }
                }

                /// Whether every counter in this block is zero.
                pub fn is_empty(&self) -> bool {
                    self.arith.is_empty()
                }
            }

            impl core::fmt::Display for OpsCount {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(f, "{}", self.arith)
                }
            }

            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct OpsCounts {
                $(pub [< $elems:lower >]: OpsCount),*
            }

            impl core::fmt::Display for OpsCounts {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let mut any = false;
                    $(
                        if !self.[< $elems:lower >].is_empty() {
                            writeln!(f, "    {}: {}", stringify!($elems), self.[< $elems:lower >])?;
                            any = true;
                        }
                    )*
                    if !any {
                        writeln!(f, "    (none)")?;
                    }
                    Ok(())
                }
            }
        }
    }
}

ops_counts!(
    ariths: [Add, SaturatingAdd, Fma, Sub, SaturatingSub, Mul, Div, Abs, Exp, Log, Log1p, Expm1, Cos, Sin, Tan, Tanh, Sinh, Cosh, ArcCos, ArcSin, ArcTan, ArcSinh, ArcCosh, ArcTanh, Degrees, Radians, ArcTan2, Powf, Powi, Hypot, Rhypot, Sqrt, InverseSqrt, Round, Floor, Ceil, Trunc, Erf, Recip, Clamp, Neg, Max, Min, Rem, ModFloor, Magnitude, Normalize, Dot, MulHi, VectorSum],
    elems: [Bool, E2M1, E2M3, E3M2, E4M3, E5M2, UE8M0, F16, BF16, Flex32, F32, TF32, F64, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
);

impl OpsCounts {
    pub const LEN: usize = size_of::<Self>() / size_of::<u32>();

    pub fn stored_type() -> Type {
        Type::atomic(Type::scalar(ElemType::UInt(UIntKind::U32)))
    }

    pub fn elem_index(elem: &ElemType) -> usize {
        match elem {
            ElemType::Bool => 0,
            ElemType::Float(FloatKind::E2M1) => 1,
            ElemType::Float(FloatKind::E2M3) => 2,
            ElemType::Float(FloatKind::E3M2) => 3,
            ElemType::Float(FloatKind::E4M3) => 4,
            ElemType::Float(FloatKind::E5M2) => 5,
            ElemType::Float(FloatKind::UE8M0) => 6,
            ElemType::Float(FloatKind::F16) => 7,
            ElemType::Float(FloatKind::BF16) => 8,
            ElemType::Float(FloatKind::Flex32) => 9,
            ElemType::Float(FloatKind::F32) => 10,
            ElemType::Float(FloatKind::TF32) => 11,
            ElemType::Float(FloatKind::F64) => 12,
            ElemType::Int(IntKind::I8) => 13,
            ElemType::Int(IntKind::I16) => 14,
            ElemType::Int(IntKind::I32) => 15,
            ElemType::Int(IntKind::I64) => 16,
            ElemType::UInt(UIntKind::U8) => 17,
            ElemType::UInt(UIntKind::U16) => 18,
            ElemType::UInt(UIntKind::U32) => 19,
            ElemType::UInt(UIntKind::U64) => 20,
        }
    }

    pub fn offset(elem: &ElemType, op: &Operation) -> Option<usize> {
        OpsCount::offset(op).map(|within| Self::elem_index(elem) * OpsCount::LEN + within)
    }

    pub fn tracked_name(elem: &ElemType, op: &Operation) -> String {
        format!("{:?}_{:?}", elem.to_string(), op.to_string())
    }
}

#[derive(Debug)]
pub struct OpsCountsProcessor {
    ops_counts: Value,
}

impl OpsCountsProcessor {
    /// Create a processor that increments the given global atomic `counter` array.
    pub fn new(ops_counts: Value) -> Self {
        Self { ops_counts }
    }

    /// Append `counter[slot] += amount` to the processing instruction stream.
    fn emit_increment(&self, processing: &mut ScopeProcessing, slot: u32, amount: u32) {
        // Build the increment in a throwaway scope that shares the global state (allocator, type
        // maps, ...) so the freshly created values don't collide with existing ones. Profiling is
        // disabled on this scope so its own `process` call doesn't recurse into FLOP counting.
        let scope = Scope::root(false, false).with_global_state(processing.global_state.clone());

        let u32_ty = Type::scalar(ElemType::UInt(UIntKind::U32));

        // `&counter[slot]` -> pointer to the atomic element for this op.
        let elem_ptr = scope.create_value(Type::pointer(
            self.ops_counts.value_type(),
            self.ops_counts.address_space(),
        ));
        scope.register(Instruction::new(
            Memory::Index(IndexOperands {
                list: self.ops_counts,
                index: Value::constant(slot.into(), u32_ty),
                unroll_factor: 1,
                checked: false,
            }),
            elem_ptr,
        ));

        // `atomicAdd(&counter[slot], amount)`. The returned old value is unused.
        let old = scope.create_value(u32_ty);
        scope.register(Instruction::new(
            AtomicOp::Add(AtomicBinaryOperands {
                ptr: elem_ptr,
                value: Value::constant(amount.into(), u32_ty),
            }),
            old,
        ));

        let tmp = scope.process([]);
        processing.instructions.extend(tmp.instructions);
    }
}

impl Processor for OpsCountsProcessor {
    fn transform(&self, mut processing: ScopeProcessing) -> ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            let metric = match &instruction.operation {
                Operation::Arithmetic(_) => {
                    OpsCounts::offset(&instruction.out().elem_type(), &instruction.operation)
                        .map(|slot| (slot as u32, instruction.out().vector_size() as u32))
                }
                _ => None,
            };

            processing.instructions.push(instruction);

            if let Some((slot, amount)) = metric {
                self.emit_increment(&mut processing, slot, amount);
            }
        }

        processing
    }
}

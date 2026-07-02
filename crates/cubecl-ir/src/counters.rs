use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use paste::paste;

use crate::AddressSpace;
use crate::{
    ArithmeticOpCode, ElemType, FloatKind, IntKind, Memory, MemoryOpCode, Operation,
    OperationReflect, Type, UIntKind, Value,
};

fn display_u32(value: u32) -> String {
    let s = value.to_string();
    s.as_bytes()
        .rchunks(3)
        .rev()
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(derive_new::new)]
pub struct CountInfo {
    pub slot: u32,
    pub amount: u32,
}

pub struct OpsCounter;

impl OpsCounter {
    pub fn count(operation: &Operation, out: Option<&Value>) -> Option<CountInfo> {
        match operation {
            Operation::Arithmetic(_) => {
                let out = out?;
                OpsCounts::offset(&out.elem_type(), operation)
                    .map(|index| CountInfo::new(index as u32, out.vector_size() as u32))
            }
            Operation::Memory(memory) => {
                let (value, target_address_spaces) = match memory {
                    Memory::Index(_) => {
                        let out = out?;
                        (out, std::vec![out.address_space()])
                    }
                    Memory::Load(variable) => (variable, std::vec![variable.address_space()]),
                    Memory::Store(op) => (&op.value, std::vec![op.ptr.address_space()]),
                    Memory::CopyMemory(op) => (
                        &op.source,
                        std::vec![op.source.address_space(), op.target.address_space()],
                    ),
                };

                if target_address_spaces.iter().any(|addr_space| {
                    !matches!(addr_space, AddressSpace::Local | AddressSpace::Shared)
                }) {
                    OpsCounts::offset(&value.elem_type(), operation).map(|index| {
                        CountInfo::new(
                            index as u32,
                            (value.vector_size() * value.elem_type().size()) as u32,
                        )
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

macro_rules! ops_count {
    ({$($op:ident),*}, $name:ident, $unit:literal) => {
        paste! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct [< $name OpsCount >] {
                $(pub [< $op:lower >]: u32),*
            }

            impl [< $name OpsCount >] {
                pub const LEN: usize = size_of::<Self>() / size_of::<u32>();

                pub fn offset(op_code: &[< $name OpCode >]) -> Option<usize> {
                    enum OpIndex {
                        $($op),*
                    }

                    Some(match op_code {
                        $([< $name OpCode >]::$op => OpIndex::$op as usize),*
                    })
                }

                pub fn is_empty(&self) -> bool {
                    true $(&& self.[< $op:lower >] == 0)*
                }

                pub const KIND: &'static str = stringify!($name);
                pub const UNIT: &'static str = $unit;

                pub fn entries(&self) -> Vec<(&'static str, u32)> {
                    let mut entries = Vec::new();
                    $(
                        if self.[< $op:lower >] != 0 {
                            entries.push((stringify!($op), self.[< $op:lower >]));
                        }
                    )*
                    entries
                }
            }

            impl core::fmt::Display for [< $name OpsCount >] {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let mut first = true;
                    $(
                        if self.[< $op:lower >] != 0 {
                            if !first {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}: {}{}", stringify!($op), display_u32(self.[< $op:lower >]), $unit)?;
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
       memories  : [$($memory:ident),*],
       elems: [$($elems:ty),*]
    ) => {
    paste! {
            ops_count!({$($arith),*}, Arithmetic, "ops");
            ops_count!({$($memory),*}, Memory, "bytes");

            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct OpsCount {
                pub arith: ArithmeticOpsCount,
                pub memory: MemoryOpsCount,
            }

            impl OpsCount {
                pub const LEN: usize = size_of::<Self>() / size_of::<u32>();

                pub fn offset(op: &Operation) -> Option<usize> {
                    match op {
                        Operation::Arithmetic(arith) => {
                            let base = core::mem::offset_of!(OpsCount, arith) / size_of::<u32>();
                            ArithmeticOpsCount::offset(&arith.op_code()).map(|o| base + o)
                        }
                        Operation::Memory(memory) => {
                            let base = core::mem::offset_of!(OpsCount, memory) / size_of::<u32>();
                            MemoryOpsCount::offset(&memory.op_code()).map(|o| base + o)
                        }
                        _ => None,
                    }
                }

                pub fn is_empty(&self) -> bool {
                    self.arith.is_empty() && self.memory.is_empty()
                }

                pub fn entries(&self) -> Vec<(&'static str, &'static str, &'static str, u32)> {
                    let mut entries = Vec::new();
                    for (op, count) in self.arith.entries() {
                        entries.push((ArithmeticOpsCount::KIND, ArithmeticOpsCount::UNIT, op, count));
                    }
                    for (op, count) in self.memory.entries() {
                        entries.push((MemoryOpsCount::KIND, MemoryOpsCount::UNIT, op, count));
                    }
                    entries
                }
            }

            impl core::fmt::Display for OpsCount {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(f, "{}", self.arith)?;
                    write!(f, ", {}", self.memory)?;
                    Ok(())
                }
            }

            #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
            #[repr(C)]
            pub struct OpsCounts {
                $(pub [< $elems:lower >]: OpsCount),*
            }

            impl core::fmt::Display for OpsCounts {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    let mut entries = Vec::new();
                    $(
                        for (kind, unit, op, count) in self.[< $elems:lower >].entries() {
                            entries.push((
                                stringify!($elems),
                                kind,
                                op,
                                display_u32(count),
                                unit
                            ));
                        }
                    )*

                    if entries.is_empty() {
                        return writeln!(f, "    (none)");
                    }

                    let max_op_len = entries.iter().map(|e| e.2.len()).max().unwrap_or(0);
                    let max_count_len = entries.iter().map(|e| e.3.len()).max().unwrap_or(0);

                    let mut current_elem = "";
                    let mut current_kind = "";
                    let mut is_last_kind = false;

                    for i in 0..entries.len() {
                        let (elem, kind, op, ref count_str, unit) = entries[i];

                        if elem != current_elem {
                            writeln!(f, "{}", elem)?;
                            current_elem = elem;
                            current_kind = "";
                        }

                        if kind != current_kind {
                            is_last_kind = !entries[i + 1..]
                                .iter()
                                .any(|e| e.0 == elem && e.1 != kind);

                            let kind_prefix = if is_last_kind { "└── " } else { "├── " };
                            writeln!(f, "{}{}", kind_prefix, kind)?;
                            current_kind = kind;
                        }

                        let is_last_op = i + 1 == entries.len()
                            || entries[i + 1].0 != elem
                            || entries[i + 1].1 != kind;

                        let kind_indent = if is_last_kind { "    " } else { "│   " };
                        let op_prefix = if is_last_op { "└── " } else { "├── " };

                        writeln!(
                            f,
                            "{}{}{:<width$} {:>count_width$} {}",
                            kind_indent,
                            op_prefix,
                            format!("{}:", op),
                            count_str,
                            unit,
                            width = max_op_len + 1,
                            count_width = max_count_len
                        )?;
                    }

                    Ok(())
                }
            }
        }
    }
}

ops_counts!(
    ariths: [Add, SaturatingAdd, Fma, Sub, SaturatingSub, Mul, Div, Abs, Exp, Log, Log1p, Expm1, Cos, Sin, Tan, Tanh, Sinh, Cosh, ArcCos, ArcSin, ArcTan, ArcSinh, ArcCosh, ArcTanh, Degrees, Radians, ArcTan2, Powf, Powi, Hypot, Rhypot, Sqrt, InverseSqrt, Round, Floor, Ceil, Trunc, Erf, Recip, Clamp, Neg, Max, Min, Rem, ModFloor, Magnitude, Normalize, Dot, MulHi, VectorSum],
    memories: [Index, Load, Store, CopyMemory],
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
}

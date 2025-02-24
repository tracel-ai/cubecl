use half::{bf16, f16};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    Atomic(AtomicKind),
    Bool,
    // signed
    I8,
    I16,
    I32,
    I64,
    // unsigned
    U8,
    U16,
    U32,
    U64,
    // float
    BF16,
    F16,
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum AtomicKind {
    // TODO: Add atomic bool support at the low ir
    Bool,
    // signed
    I32,
    I64,
    // unsigned
    U32,
    U64,
    // float
    F32,
}

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Elem::Bool | Elem::Atomic(AtomicKind::Bool) => core::mem::size_of::<bool>(),
            // signed
            Elem::I8 => core::mem::size_of::<i8>(),
            Elem::I16 => core::mem::size_of::<i16>(),
            Elem::I32 | Elem::Atomic(AtomicKind::I32) => core::mem::size_of::<i32>(),
            Elem::I64 | Elem::Atomic(AtomicKind::I64) => core::mem::size_of::<i64>(),
            // unsigned
            Elem::U8 => core::mem::size_of::<u8>(),
            Elem::U16 => core::mem::size_of::<u16>(),
            Elem::U32 | Elem::Atomic(AtomicKind::U32) => core::mem::size_of::<u32>(),
            Elem::U64 | Elem::Atomic(AtomicKind::U64) => core::mem::size_of::<u64>(),
            // float
            Elem::BF16 => core::mem::size_of::<bf16>(),
            Elem::F16 => core::mem::size_of::<f16>(),
            Elem::F32 | Elem::Atomic(AtomicKind::F32) => core::mem::size_of::<f32>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        match self {
            Self::Atomic(_) => true,
            _ => false,
        }
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atomic(kind) => write!(f, "atomic_{kind}"),
            Self::Bool => f.write_str("bool"),
            // signed
            Self::I8 => f.write_str("char"),
            Self::I16 => f.write_str("short"),
            Self::I32 => f.write_str("int"),
            Self::I64 => f.write_str("long"),
            // unsigned
            Self::U8 => f.write_str("uchar"),
            Self::U16 => f.write_str("ushort"),
            Self::U32 => f.write_str("uint"),
            Self::U64 => f.write_str("ulong"),
            // float
            Self::BF16 => f.write_str("bfloat"),
            Self::F16 => f.write_str("half"),
            Self::F32 => f.write_str("float"),
        }
    }
}

impl Display for AtomicKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => Elem::Bool.fmt(f),
            // signed
            Self::I32 => Elem::I32.fmt(f),
            Self::I64 => Elem::I64.fmt(f),
            // unsigned
            Self::U32 => Elem::U32.fmt(f),
            Self::U64 => Elem::U64.fmt(f),
            // float
            Self::F32 => Elem::F32.fmt(f),
        }
    }
}

use cubecl_common::{e2m1x2, e3m2, e5m2};
use cubecl_core::tf32;
use half::{bf16, f16};
use std::fmt::Display;

use super::Dialect;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum Elem<D: Dialect> {
    TF32,
    F32,
    F64,
    F16,
    F16x2,
    BF16,
    BF16x2,
    FP4(FP4Kind),
    FP4x2(FP4Kind),
    FP6(FP6Kind),
    FP6x2(FP6Kind),
    FP8(FP8Kind),
    FP8x2(FP8Kind),
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Atomic(AtomicKind<D>),
    _Dialect(std::marker::PhantomData<D>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FP4Kind {
    E2M1,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FP6Kind {
    E2M3,
    E3M2,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum FP8Kind {
    E4M3,
    E5M2,
    UE8M0,
}

impl Display for FP4Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            FP4Kind::E2M1 => "e2m1",
        };
        f.write_str(name)
    }
}

impl Display for FP6Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            FP6Kind::E2M3 => "e2m3",
            FP6Kind::E3M2 => "e3m2",
        };
        f.write_str(name)
    }
}

impl Display for FP8Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            FP8Kind::E4M3 => "e4m3",
            FP8Kind::E5M2 => "e5m2",
            FP8Kind::UE8M0 => "e8m0",
        };
        f.write_str(name)
    }
}

impl<D: Dialect> Display for Elem<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_elem(f, self, false)
    }
}

impl<D: Dialect> Elem<D> {
    pub const fn size(&self) -> usize {
        match self {
            Elem::FP4(_) => panic!("Can't get byte size of sub-byte type"),
            Elem::FP4x2(_) => core::mem::size_of::<e2m1x2>(),
            Elem::FP6(_) => core::mem::size_of::<e3m2>(),
            Elem::FP6x2(_) => 2 * core::mem::size_of::<e3m2>(),
            Elem::FP8(_) => core::mem::size_of::<e5m2>(),
            Elem::FP8x2(_) => 2 * core::mem::size_of::<e5m2>(),
            Elem::F16 => core::mem::size_of::<f16>(),
            Elem::F16x2 => 2 * core::mem::size_of::<f16>(),
            Elem::BF16x2 => 2 * core::mem::size_of::<bf16>(),
            Elem::BF16 => core::mem::size_of::<bf16>(),
            Elem::TF32 => core::mem::size_of::<tf32>(),
            Elem::F32 => core::mem::size_of::<f32>(),
            Elem::F64 => core::mem::size_of::<f64>(),
            Elem::I8 => core::mem::size_of::<i8>(),
            Elem::I16 => core::mem::size_of::<i16>(),
            Elem::I32 => core::mem::size_of::<i32>(),
            Elem::I64 => core::mem::size_of::<i64>(),
            Elem::U8 => core::mem::size_of::<u8>(),
            Elem::U16 => core::mem::size_of::<u16>(),
            Elem::U32 => core::mem::size_of::<u32>(),
            Elem::U64 => core::mem::size_of::<u64>(),
            Elem::Bool => core::mem::size_of::<bool>(),
            Elem::Atomic(AtomicKind::I32) => core::mem::size_of::<i32>(),
            Elem::Atomic(AtomicKind::I64) => core::mem::size_of::<i64>(),
            Elem::Atomic(AtomicKind::U32) => core::mem::size_of::<u32>(),
            Elem::Atomic(AtomicKind::U64) => core::mem::size_of::<u64>(),
            Elem::Atomic(AtomicKind::F16) => core::mem::size_of::<f16>(),
            Elem::Atomic(AtomicKind::BF16) => core::mem::size_of::<bf16>(),
            Elem::Atomic(AtomicKind::F32) => core::mem::size_of::<f32>(),
            Elem::Atomic(AtomicKind::F64) => core::mem::size_of::<f64>(),
            Elem::Atomic(AtomicKind::_Dialect(_)) => 0,
            Elem::_Dialect(_) => 0,
        }
    }

    pub const fn size_bits(&self) -> usize {
        match self {
            Elem::FP4(_) => 4,
            other => other.size() * 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum AtomicKind<D: Dialect> {
    I32,
    I64,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    /// Required to construct the inner `Elem` of the atomic value
    _Dialect(std::marker::PhantomData<D>),
}

impl<D: Dialect> Display for AtomicKind<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_atomic_kind(f, self)
    }
}

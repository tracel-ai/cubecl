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
    F162,
    BF16,
    BF162,
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
        match self {
            Self::I32 => Elem::<D>::I32.fmt(f),
            Self::I64 => Elem::<D>::I64.fmt(f),
            Self::U32 => Elem::<D>::U32.fmt(f),
            Self::U64 => Elem::<D>::U64.fmt(f),
            Self::F16 => Elem::<D>::F16.fmt(f),
            Self::BF16 => Elem::<D>::BF16.fmt(f),
            Self::F32 => Elem::<D>::F32.fmt(f),
            Self::F64 => Elem::<D>::F64.fmt(f),
            Self::_Dialect(_) => Ok(()),
        }
    }
}

impl<D: Dialect> Display for Elem<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_elem(f, &self)
    }
}

impl<D: Dialect> Elem<D> {
    pub const fn size(&self) -> usize {
        match self {
            Elem::F16 => core::mem::size_of::<f16>(),
            Elem::F162 => 2 * core::mem::size_of::<f16>(),
            Elem::BF162 => 2 * core::mem::size_of::<bf16>(),
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
}

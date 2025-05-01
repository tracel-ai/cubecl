use cubecl_core::{flex32, prelude::Numeric};

/// Precision used for the reduction.
pub trait ReducePrecision {
    /// Precision used for the input tensor.
    type EI: Numeric;
    /// Precision used for the accumulation.
    type EA: Numeric;
}

impl<EI: Numeric, EA: Numeric> ReducePrecision for (EI, EA) {
    type EI = EI;
    type EA = EA;
}

// The below implementations are suggestion for reduction that can accumulate precision errors like
// summations.

impl ReducePrecision for f64 {
    type EI = f64;
    type EA = f64;
}

impl ReducePrecision for f32 {
    type EI = f32;
    type EA = f32;
}

impl ReducePrecision for flex32 {
    type EI = f32;
    type EA = f32;
}

impl ReducePrecision for half::f16 {
    type EI = half::f16;
    type EA = f32;
}

impl ReducePrecision for half::bf16 {
    type EI = half::bf16;
    type EA = f32;
}

impl ReducePrecision for i64 {
    type EI = i64;
    type EA = i64;
}

impl ReducePrecision for i32 {
    type EI = i32;
    type EA = i32;
}

impl ReducePrecision for i16 {
    type EI = i16;
    type EA = i32;
}

impl ReducePrecision for i8 {
    type EI = i8;
    type EA = i32;
}

impl ReducePrecision for u64 {
    type EI = u64;
    type EA = u64;
}

impl ReducePrecision for u32 {
    type EI = u32;
    type EA = u32;
}

impl ReducePrecision for u16 {
    type EI = u16;
    type EA = u32;
}

impl ReducePrecision for u8 {
    type EI = u8;
    type EA = u32;
}

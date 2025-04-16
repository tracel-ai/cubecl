use cubecl_core::prelude::Numeric;

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

impl ReducePrecision for f32 {
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

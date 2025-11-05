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

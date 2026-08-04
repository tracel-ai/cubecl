use cubecl_common::quant::scheme::QuantLevel;
use cubecl_core::prelude::Scalar;

/// Run an arbitrary function with the quantization types from the scheme.
/// Useful when concrete types aren't available.
pub trait RunWithQuantType {
    type Output;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output;
}

/// Panic for a level these kernels cannot reconstruct.
///
/// They apply one scale per value and never consult the level, so a per-tensor scale would be
/// dropped and every value would come back short by that factor. Levels are matched exhaustively so
/// a new one has to make a support decision here rather than inherit silence.
pub fn assert_level_supported(level: QuantLevel) {
    match level {
        QuantLevel::Tensor | QuantLevel::Block(_) => {}
        QuantLevel::BlockTensor { .. } => {
            panic!("two-level quantization is not supported by these kernels, got {level:?}")
        }
    }
}

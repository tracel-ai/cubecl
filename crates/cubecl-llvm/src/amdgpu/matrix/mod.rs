//! AMDGPU matrix operations.

mod base;
mod quantized;

pub use base::{
    MatrixElemUnsupported, MatrixRelayoutUnsupported, MatrixShapeUnsupported,
    MatrixTileMoveUnsupported, WmmaForm,
};
pub(crate) use base::{
    cast, fill, fragment_ty, ld_matrix, load, mma_manual, multiply_accumulate, set_matrix_core,
    st_matrix, store,
};
pub(crate) use quantized::mma_quantized;
pub use quantized::{QuantizedShapeUnsupported, quantized_configs};

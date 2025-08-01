use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::{MatmulIdent, MatmulPrecision};

/// Store the quantization meta-parameters.
/// For now, we only support symmetric quantization,
/// thus we only store the scaling.
#[derive(CubeType, Clone, Copy)]
pub struct Quantization<MP: MatmulPrecision> {
    pub scaling_lhs: MP::ES,
    pub scaling_rhs: MP::ES,
}

#[cube]
impl<MP: MatmulPrecision> Quantization<MP> {
    pub fn dequantize(&self, line: Line<MP::EI>, #[comptime] ident: MatmulIdent) -> Line<MP::ES> {
        match ident {
            MatmulIdent::Lhs => Line::<MP::ES>::new(self.scaling_lhs) * Line::cast_from(line),
            MatmulIdent::Rhs => Line::<MP::ES>::new(self.scaling_rhs) * Line::cast_from(line),
            MatmulIdent::Out => comptime!(unreachable!()),
        }
    }
}

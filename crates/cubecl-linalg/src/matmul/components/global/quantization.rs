use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::matmul::components::MatmulPrecision;

/// Store the quantization meta-parameters.
/// For now, we only support symmetric quantization,
/// thus we only store the scaling.
#[derive(CubeType, Clone, Copy)]
pub struct Quantization<MP: MatmulPrecision> {
    // I use MP instead of simply ES to be future proof.
    pub scaling: MP::ES,
}

#[cube]
impl<MP: MatmulPrecision> Quantization<MP> {
    pub fn dequantize(&self, line: Line<MP::EI>) -> Line<MP::ES> {
        Line::<MP::ES>::new(self.scaling) * Line::cast_from(line)
    }
}

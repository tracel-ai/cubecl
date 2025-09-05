use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::View;

use crate::scheme::{QuantLevel, QuantScheme};

/// Quantization parameters.
#[derive(CubeLaunch, CubeType)]
pub struct QParams {
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    pub num_quants: u32,
}

#[cube]
impl QParams {
    /// Create a new quantization parameters instance.
    pub fn new(#[comptime] scheme: QuantScheme) -> Self {
        let num_quants = comptime!(scheme.num_quants() as u32);
        QParams { scheme, num_quants }
    }

    /// Get the quantization parameters values.
    pub fn scale<F: Float>(&self, scale_tensor: &View<Line<F>, u32>, value_pos: u32) -> F {
        match comptime!(self.scheme) {
            // Symmetric quantization only contains the scaling factor as the last element
            QuantScheme {
                level: QuantLevel::Tensor,
                ..
            } => scale_tensor[0][0],
            QuantScheme {
                level: QuantLevel::Block(block_size),
                ..
            } => {
                // The input position is `num_quants` smaller because it acts as vectorize with a line
                // size, but the scales don't have any line size.
                let position = value_pos * self.num_quants;
                scale_tensor[position / comptime! {block_size as u32}][0]
            }
        }
    }
}

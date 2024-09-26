use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::interface::matmul_4_tile::base::TileMatmul;

#[cube]
/// Handles the matrix multiplication of two memory chunks
///
/// Computes A·B where
/// - A has shape [b_m, b_k]
/// - B has shape [b_k, b_n]
///
/// Responsibilities:
/// - Know where and in which format take its data
/// - Choose the best way to dispatch planes
pub trait ChunkMatmul {
    type TileMatmul: TileMatmul;

    // SMEM
    type Input: CubeType;
    // Accumulators within planes
    type Accumulator: CubeType;

    fn execute(lhs: &Self::Input, rhs: &Self::Input, acc: &mut Self::Accumulator);
}

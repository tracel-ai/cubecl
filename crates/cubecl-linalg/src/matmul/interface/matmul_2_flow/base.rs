use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::interface::matmul_3_chunk::base::ChunkMatmul;

#[cube]
/// Handles the matrix multiplication of two flows of memory chunks
///
/// Computes A·B where
/// - A has shape [b_m, k]
/// - B has shape [k, b_n]
///
/// Responsibilities:
/// - Loop over k [or partially if split involved]
/// - Accumulate
/// - Load to SMEM, perhaps with different planes than those who compute/accumulate
/// - Check bounds
pub trait FlowMatmul {
    type ChunkMatmul: ChunkMatmul;

    // GMEM, but offseted in row or column
    // Different type for transposed (?)
    type Input: CubeType;
    // Accumulators within planes, but not part of execute interface
    type Accumulator: CubeType;
    // GMEM, offseted in both row and column
    type Output: CubeType;

    fn execute(lhs: &Self::Input, rhs: &Self::Input, out: &mut Self::Output);
}

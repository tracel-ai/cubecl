use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// Handles the matrix multiplication of two tiles
/// Should match hardware specifications
///
/// Computes A·B where
/// - A has shape [t_m, t_k]
/// - B has shape [t_k, t_n]
///
/// Responsibilities:
/// - Perform actual computations of multiply and add
pub trait TileMatmul {
    // For cmma: plane, for tiling2d: unit
    type Performer: CubeType;
    // Fragments a, b
    type Input: CubeType;
    // Fragment accumulator
    type Accumulator: CubeType;
    type Config: CubeType;

    fn execute(
        lhs: &Self::Input,
        rhs: &Self::Input,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );
}

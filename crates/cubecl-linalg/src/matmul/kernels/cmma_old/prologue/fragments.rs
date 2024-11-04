use cubecl_core::cube;
use cubecl_core::{self as cubecl, prelude::*};

use super::super::config::ComptimeCmmaInfo;

#[derive(CubeType)]
pub(crate) struct Fragments<F: Float, FC: Float> {
    pub accumulators: Sequence<cmma::Matrix<F>>,
    pub lhs: cmma::Matrix<FC>,
    pub rhs: cmma::Matrix<FC>,
}

#[cube]
pub(crate) fn make_fragments<F: Float, FC: Float>(
    #[comptime] config: ComptimeCmmaInfo,
) -> Fragments<F, FC> {
    let num_accumulators = config.num_accumulators;
    let tile_size_m = config.tile_size_m;
    let tile_size_n = config.tile_size_n;
    let tile_size_k = config.tile_size_k;
    let mut accumulators = Sequence::<cmma::Matrix<F>>::new();

    #[unroll]
    for _ in 0..num_accumulators {
        let acc = cmma::Matrix::<F>::from_value(
            cmma::MatrixIdent::Accumulator,
            tile_size_m,
            tile_size_n,
            tile_size_k,
            cmma::MatrixLayout::Undefined,
            F::new(0.0),
        );

        accumulators.push(acc);
    }

    // Safety: these are always loaded before being used.
    let lhs = unsafe {
        cmma::Matrix::<FC>::uninitialized(
            cmma::MatrixIdent::A,
            tile_size_m,
            tile_size_n,
            tile_size_k,
            cmma::MatrixLayout::RowMajor,
        )
    };

    // Safety: these are always loaded before being used.
    let rhs = unsafe {
        cmma::Matrix::<FC>::uninitialized(
            cmma::MatrixIdent::B,
            tile_size_m,
            tile_size_n,
            tile_size_k,
            cmma::MatrixLayout::RowMajor,
        )
    };

    Fragments::<F, FC> {
        accumulators,
        lhs,
        rhs,
    }
}

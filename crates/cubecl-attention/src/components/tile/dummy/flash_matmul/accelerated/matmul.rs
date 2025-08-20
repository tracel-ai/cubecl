use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use cubecl_matmul::components::tile::Tile;

use crate::components::FlashIdent;
use crate::components::tile::dummy::accelerated::AcceleratedFlashMatmulConfig;
use crate::components::tile::dummy::{FlashMatmul, FlashMatmulConfig as _, FlashPrecision};

/// Performs two matmuls with fragment reuse for key/value and score/prob
pub struct AcceleratedFlashMatmul;

#[cube]
impl<FP: FlashPrecision> FlashMatmul<FP> for AcceleratedFlashMatmul {
    type Config = AcceleratedFlashMatmulConfig;
    type Query = cmma::Matrix<FP::Q>;
    type KeyValue = cmma::Matrix<FP::KV>;
    type ScoreProb = cmma::Matrix<FP::SP>;
    type Accumulator = cmma::Matrix<FP::A>;

    fn score_matmul(lhs: &Self::Query, rhs: &Self::KeyValue, out: &mut Self::ScoreProb) {
        cmma::execute::<FP::Q, FP::KV, FP::SP, FP::SP>(lhs, rhs, out, out);
    }

    fn value_matmul(lhs: &Self::ScoreProb, rhs: &Self::KeyValue, out: &mut Self::Accumulator) {
        cmma::execute::<FP::SP, FP::KV, FP::A, FP::A>(lhs, rhs, out, out);
    }

    fn allocate_fill_query<EI: Numeric>(
        tile: &Tile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query {
        let (slice, stride) = tile.as_unlined(config.stage_line_size(FlashIdent::Query));
        let size = config.tile_size();

        if config.cast_query() {
            let query = unsafe {
                cmma::Matrix::<FP::Q>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m(),
                    size.n(),
                    size.k(),
                    cmma::MatrixLayout::RowMajor,
                )
            };

            cmma::load(&query, &slice, stride);
            query
        } else {
            let tmp = unsafe {
                cmma::Matrix::<EI>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m(),
                    size.n(),
                    size.k(),
                    cmma::MatrixLayout::RowMajor,
                )
            };

            cmma::load(&tmp, &slice, stride);
            cmma::cast::<EI, FP::Q>(&tmp)
        }
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.tile_size();
        unsafe {
            cmma::Matrix::<FP::KV>::uninitialized(
                cmma::MatrixIdent::B,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn fill_rhs<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined(config.stage_line_size(FlashIdent::Query));
        cmma::load(rhs, &slice, stride);
    }

    fn allocate_score_prob(#[comptime] config: Self::Config) -> Self::ScoreProb {
        let size = config.tile_size();
        unsafe {
            cmma::Matrix::<FP::SP>::uninitialized(
                cmma::MatrixIdent::Accumulator, // TODO and A
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn zero_score_prob(acc: &mut Self::ScoreProb) {
        cmma::fill(acc, FP::SP::from_int(0));
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.tile_size();
        unsafe {
            cmma::Matrix::<FP::A>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator) {
        cmma::fill(acc, FP::A::from_int(0));
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<FP::A, E>(out);
        cmma::store(
            slice,
            &acc,
            config.tile_size().n(),
            cmma::MatrixLayout::RowMajor,
        );
    }

    fn tmp_fill_accumulator(
        tile: &Tile<FP::A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined(config.stage_line_size(FlashIdent::Out));
        cmma::load_with_layout(acc, &slice, stride, cmma::MatrixLayout::RowMajor);
    }
    fn tmp_fill_prob(
        tile: &Tile<FP::SP>,
        prob: &mut Self::ScoreProb,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined(1u32);
        cmma::load_with_layout(prob, &slice, stride, cmma::MatrixLayout::RowMajor);
    }
    fn tmp_write_score_prob<E: Numeric>(
        out: &Self::ScoreProb,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let to_store = cmma::cast::<FP::SP, E>(out);
        cmma::store(
            slice,
            &to_store,
            config.tile_size().n(),
            cmma::MatrixLayout::RowMajor,
        );
    }
}

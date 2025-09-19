use std::cmp::max;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::Tile;

use crate::components::tile::dummy::dummy_register::DummyRegisterFlashMatmulConfig;
use crate::components::tile::dummy::{FlashMatmul, FlashMatmulConfig as _, FlashPrecision};

/// Dummy FlashMatmul implementation using simple arrays
/// Only lane 0 performs computations, other lanes idle
pub struct DummyRegisterFlashMatmul;

#[cube]
impl<FP: FlashPrecision> FlashMatmul<FP> for DummyRegisterFlashMatmul {
    type Config = DummyRegisterFlashMatmulConfig;

    type Query = Array<FP::Q>;
    type KeyValue = Array<FP::KV>;
    type ScoreProb = Array<FP::SP>;
    type Accumulator = Array<FP::A>;

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::ScoreProb,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_score_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = FP::SP::from_int(0);
                    for ki in 0..k {
                        let lhs_val = lhs[i * k + ki];
                        let rhs_val = rhs[ki * n + j];
                        sum += FP::SP::cast_from(lhs_val) * FP::SP::cast_from(rhs_val);
                    }
                    out[i * n + j] += sum;
                }
            }
        }

        sync_cube();
    }

    fn value_matmul(
        lhs: &Self::ScoreProb,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_value_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = FP::A::from_int(0);
                    for ki in 0..k {
                        let lhs_val = lhs[i * k + ki];
                        let rhs_val = rhs[ki * n + j];
                        sum += FP::A::cast_from(lhs_val) * FP::A::cast_from(rhs_val);
                    }
                    out[i * n + j] += sum;
                }
            }
        }

        sync_cube();
    }

    fn allocate_fill_query<EI: Numeric>(
        tile: &Tile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query {
        let seq_q = config.attention_tile_size().seq_q;
        let head_dim = config.attention_tile_size().head_dim;

        let mut query = Array::<FP::Q>::new(seq_q * head_dim);

        if UNIT_POS == 0 {
            // Only lane 0 fills the data
            for q in 0..seq_q {
                for hd in 0..head_dim {
                    query[q * head_dim + hd] = FP::Q::cast_from(tile.get_line(q, hd));
                }
            }
        }

        sync_cube();
        query
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<FP::KV>::new(comptime!(max(
            config.attention_tile_size().key_size(),
            config.attention_tile_size().value_size(),
        )))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<FP::KV>::new(config.attention_tile_size().key_size())
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<FP::KV>::new(config.attention_tile_size().value_size())
    }

    fn fill_key_value<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let size = config.attention_tile_size().key_size();
            for i in 0..size {
                rhs[i] = FP::KV::cast_from(tile.as_unlined(1u32).0[i]);
            }
        }

        sync_cube();
    }

    fn allocate_score_prob(#[comptime] config: Self::Config) -> Self::ScoreProb {
        Array::<FP::SP>::new(config.attention_tile_size().score_prob_size())
    }

    fn zero_score_prob(score_prob: &mut Self::ScoreProb, #[comptime] config: Self::Config) {
        if UNIT_POS == 0 {
            let len = config.attention_tile_size().score_prob_size();
            for i in 0..len {
                score_prob[i] = FP::SP::from_int(0);
            }
        }
        sync_cube();
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        Array::<FP::A>::new(config.attention_tile_size().accumulator_size())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        if UNIT_POS == 0 {
            let len = config.attention_tile_size().accumulator_size();
            for i in 0..len {
                acc[i] = FP::A::from_int(0);
            }
        }

        sync_cube();
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let size = config.attention_tile_size().accumulator_size();
            for i in 0..size {
                slice[i] = Line::cast_from(out[i]);
            }
        }

        sync_cube();
    }

    fn tmp_fill_accumulator(
        tile: &Tile<FP::A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let size = config.attention_tile_size().accumulator_size();
            for i in 0..size {
                acc[i] = tile.as_unlined(1u32).0[i];
            }
        }

        sync_cube();
    }

    fn tmp_fill_prob(
        tile: &Tile<FP::SP>,
        prob: &mut Self::ScoreProb,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let len = config.attention_tile_size().score_prob_size();
            for i in 0..len {
                prob[i] = tile.as_unlined(1u32).0[i];
            }
        }

        sync_cube();
    }

    fn tmp_write_score_prob(
        score_prob: &Self::ScoreProb,
        slice: &mut SliceMut<Line<FP::SP>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS == 0 {
            let size = config.attention_tile_size().score_prob_size();
            for i in 0..size {
                slice[i] = Line::cast_from(score_prob[i]);
            }
        }

        sync_cube();
    }
}

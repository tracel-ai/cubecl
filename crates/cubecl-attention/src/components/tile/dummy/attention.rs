use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use std::marker::PhantomData;

use crate::components::TileMask;
use crate::components::tile::ScaleMode;
use crate::components::tile::accumulator::AccumulatorTile as _;
use crate::components::tile::accumulator::AccumulatorTileExpand;
use crate::components::tile::dummy::DummyAccumulator;
use crate::components::tile::dummy::{DummySoftmax, FlashMatmul, FlashPrecision};
use crate::components::tile::softmax::SoftmaxTileExpand;
use crate::components::tile::{RowWise, RunningState, SoftmaxTile, TileAttention};
use crate::components::{
    AttentionPrecision,
    tile::dummy::{KeyValueFragment, QueryFragment},
};

pub struct DummyTileAttention<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    _phantom: PhantomData<(FP, FM)>,
}

#[cube]
impl<AP: AttentionPrecision, FM: FlashMatmul<AP::FlashPrecision>> TileAttention<AP>
    for DummyTileAttention<AP::FlashPrecision, FM>
{
    type Config = FM::Config;

    type QueryTile = QueryFragment<AP::FlashPrecision, FM>;
    type KeyValueTile = KeyValueFragment<AP::FlashPrecision, FM>;
    type SoftmaxTile = DummySoftmax<AP::FlashPrecision, FM>;
    type AccumulatorTile = DummyAccumulator<AP::FlashPrecision, FM>;

    fn rescale(
        acc: &mut Self::AccumulatorTile,
        prev_state: &RunningState<AP::EA>,
        #[comptime] _config: Self::Config,
    ) {
        acc.scale(&prev_state.l, ScaleMode::Divide);
    }

    fn write_results(
        acc: &Self::AccumulatorTile,
        slice: &mut SliceMut<Line<AP::EO>>,
        #[comptime] tile_config: Self::Config,
    ) {
        FM::write_results(&acc.fragment, slice, tile_config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorTile {
        Self::AccumulatorTile::new(config)
    }

    fn init_query(tile: &StridedTile<AP::EI>, #[comptime] config: Self::Config) -> Self::QueryTile {
        Self::QueryTile::new(tile, config)
    }

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValueTile {
        Self::KeyValueTile::new_key_value(config)
    }

    fn init_key(#[comptime] config: Self::Config) -> Self::KeyValueTile {
        Self::KeyValueTile::new_key(config)
    }

    fn init_value(#[comptime] config: Self::Config) -> Self::KeyValueTile {
        Self::KeyValueTile::new_value(config)
    }

    fn init_score(#[comptime] config: Self::Config) -> Self::SoftmaxTile {
        Self::SoftmaxTile::new(config)
    }

    fn fill_key<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    ) {
        FM::fill_key_value(tile, rhs.key_mut(), config);
    }

    fn fill_value<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    ) {
        FM::fill_key_value(tile, rhs.value_mut(), config);
    }

    fn zero_softmax(score: &mut Self::SoftmaxTile, #[comptime] config: Self::Config) {
        FM::zero_score_prob(&mut score.fragment, config);
    }

    fn accumulate_score(
        query: &Self::QueryTile,
        key_value: &Self::KeyValueTile,
        score_prob: &mut Self::SoftmaxTile,
        #[comptime] config: Self::Config,
    ) {
        FM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut score_prob.fragment,
            config,
        );
    }

    fn softmax(
        softmax: &mut Self::SoftmaxTile,
        mask: TileMask,
        state: &mut RunningState<AP::EA>,
        #[comptime] dk: u32,
    ) -> RowWise<AP::EA> {
        let inv_sqrt_dk = AP::EA::new(comptime!(1.0 / (dk as f32).sqrt()));

        softmax.scale_and_mask(inv_sqrt_dk, mask);

        let score_max = softmax.row_max(state.m.copy());

        softmax.to_prob(state, &score_max)
    }

    // fn score_to_prob(
    //     score_prob: &mut Self::Softmax,
    //     mask: TileMask,
    //     state: &RunningState<AP::EA>,
    //     #[comptime] dk: u32,
    // ) -> RowStats<AP::EA> {
    //     let inv_sqrt_dk = AP::EA::new(comptime!(1.0 / (dk as f32).sqrt()));

    //     score_prob.multiply_score(inv_sqrt_dk);

    //     score_prob.apply_mask(mask);

    //     let max = score_prob.row_max(state.m);

    //     score_prob.to_prob(max);
    //     let prob_row_sum = score_prob.row_sum();

    //     RowStats::<AP::EA> {
    //         m: max,
    //         prob_row_sum,
    //     }
    // }

    // fn update_state(
    //     state: &mut RunningState<AP::EA>,
    //     score_prob_row_stats: &RowStats<AP::EA>,
    // ) -> AP::EA {
    //     let prev_m = state.m;
    //     let prev_l = state.l;
    //     let new_m = score_prob_row_stats.m;

    //     let exp_m_diff = Exp::exp(prev_m - new_m);
    //     let new_l = exp_m_diff * prev_l + score_prob_row_stats.prob_row_sum;

    //     state.m = new_m;
    //     state.l = new_l;

    //     exp_m_diff
    // }

    fn accumulate_value(
        score_prob: &Self::SoftmaxTile,
        key_value: &Self::KeyValueTile,
        accumulator: &mut Self::AccumulatorTile,
        scale: &RowWise<AP::EA>,
        #[comptime] config: Self::Config,
    ) {
        accumulator.scale(scale, ScaleMode::Multiply);

        FM::value_matmul(
            &score_prob.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }

    // #[derive(CubeType)]
    // pub struct RunningState<E: Float> {
    //     pub m: E,
    //     pub l: E,
    // }

    // #[cube]
    // impl<E: Float> RunningState<E> {
    //     pub fn init() -> Self {
    //         RunningState::<E> {
    //             // TODO Neg infinity
    //             m: E::from_int(-99999999999),
    //             l: E::from_int(0),
    //         }
    //     }

    //     pub fn update(&mut self, m_new: E, l_new: E) {
    //         self.m = m_new;
    //         self.l = l_new;
    //     }
    // }
}

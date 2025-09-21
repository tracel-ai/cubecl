use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::Tile;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::tile::RowStats;
use crate::components::tile::TileAttention;
use crate::components::tile::dummy::{FlashMatmul, FlashPrecision, ScoreFragment};
use crate::components::{
    AttentionPrecision,
    tile::dummy::{AccumulatorFragment, KeyValueFragment, QueryFragment},
};

pub struct DummyTileAttention<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    _phantom: PhantomData<(FP, FM)>,
}

#[cube]
impl<AP: AttentionPrecision, FM: FlashMatmul<AP::FlashPrecision>> TileAttention<AP>
    for DummyTileAttention<AP::FlashPrecision, FM>
{
    type Config = FM::Config;

    type Query = QueryFragment<AP::FlashPrecision, FM>;
    type KeyValue = KeyValueFragment<AP::FlashPrecision, FM>;
    type ScoreProb = ScoreFragment<AP::FlashPrecision, FM>;
    type Accumulator = AccumulatorFragment<AP::FlashPrecision, FM>;

    type OutOfBoundMask = (u32, u32);

    fn rescale(
        acc: &mut Self::Accumulator,
        state: &RunningState<AP::EA>,
        #[comptime] _config: Self::Config,
    ) {
        acc.scale(AP::EA::recip(state.l));
    }

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<AP::EO>>,
        #[comptime] tile_config: Self::Config,
    ) {
        FM::write_results(&acc.fragment, slice, tile_config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        Self::Accumulator::new(config)
    }

    fn init_query(tile: &Tile<AP::EI>, #[comptime] config: Self::Config) -> Self::Query {
        Self::Query::new(tile, config)
    }

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Self::KeyValue::new_key_value(config)
    }

    fn init_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        Self::KeyValue::new_key(config)
    }

    fn init_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Self::KeyValue::new_value(config)
    }

    fn init_score(#[comptime] config: Self::Config) -> Self::ScoreProb {
        Self::ScoreProb::new(config)
    }

    fn fill_key<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        FM::fill_key_value(tile, rhs.key_mut(), config);
    }

    fn fill_value<E: Numeric>(
        tile: &Tile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        FM::fill_key_value(tile, rhs.value_mut(), config);
    }

    fn zero_score(score: &mut Self::ScoreProb, #[comptime] config: Self::Config) {
        FM::zero_score_prob(&mut score.fragment, config);
    }

    fn accumulate_score(
        query: &Self::Query,
        key_value: &Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
        #[comptime] config: Self::Config,
    ) {
        FM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut score_prob.fragment,
            config,
        );
    }

    fn score_to_prob(
        score_prob: &mut Self::ScoreProb,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        state: &RunningState<AP::EA>,
        #[comptime] dk: u32,
    ) -> RowStats<AP::EA> {
        let inv_sqrt_dk = AP::EA::new(comptime!(1.0 / (dk as f32).sqrt()));

        score_prob.multiply_score(inv_sqrt_dk);

        match out_of_bound_mask {
            CubeOption::Some(out_of_bound_mask) => score_prob.apply_mask(out_of_bound_mask),
            CubeOption::None => {}
        }

        let max = score_prob.row_max(state.m);

        score_prob.to_prob(max);
        let prob_row_sum = score_prob.row_sum();

        RowStats::<AP::EA> {
            m: max,
            prob_row_sum,
        }
    }

    fn update_state(
        state: &mut RunningState<AP::EA>,
        score_prob_row_stats: &RowStats<AP::EA>,
    ) -> AP::EA {
        let prev_m = state.m;
        let prev_l = state.l;
        let new_m = score_prob_row_stats.m;

        let exp_m_diff = Exp::exp(prev_m - new_m);
        let new_l = exp_m_diff * prev_l + score_prob_row_stats.prob_row_sum;

        state.m = new_m;
        state.l = new_l;

        exp_m_diff
    }

    fn accumulate_value(
        score_prob: &Self::ScoreProb,
        key_value: &Self::KeyValue,
        accumulator: &mut Self::Accumulator,
        scale: AP::EA,
        #[comptime] config: Self::Config,
    ) {
        accumulator.scale(scale);

        FM::value_matmul(
            &score_prob.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }
}

#[derive(CubeType)]
pub struct RunningState<E: Float> {
    pub m: E,
    pub l: E,
}

#[cube]
impl<E: Float> RunningState<E> {
    pub fn init() -> Self {
        RunningState::<E> {
            // TODO Neg infinity
            m: E::from_int(-99999999999),
            l: E::from_int(0),
        }
    }

    pub fn update(&mut self, m_new: E, l_new: E) {
        self.m = m_new;
        self.l = l_new;
    }
}

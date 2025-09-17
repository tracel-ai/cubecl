use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::GlobalWriter as _;
use cubecl_matmul::components::global::PlaneWriter;
use cubecl_matmul::components::tile::Tile;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::FlashIdent;
use crate::components::tile::RowStats;
use crate::components::tile::TileAttention;
use crate::components::tile::dummy::{
    FlashMatmul, FlashMatmulConfig, FlashPrecision, ScoreFragment,
};
use crate::components::{
    AttentionPrecision,
    global::GlobalAttentionConfig,
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

    type State = RunningState<AP::EA>;

    type Query = QueryFragment<AP::FlashPrecision, FM>;
    type KeyValue = KeyValueFragment<AP::FlashPrecision, FM>;
    type ScoreProb = ScoreFragment<AP::FlashPrecision, FM>;
    type Accumulator = AccumulatorFragment<AP::FlashPrecision, FM>;

    type OutOfBoundMask = (u32, u32);

    fn rescale(
        acc: &mut Self::Accumulator,
        state: &Self::State,
        #[comptime] _config: Self::Config,
    ) {
        acc.scale(AP::EA::recip(state.l));
    }

    fn init_state(#[comptime] _config: Self::Config) -> Self::State {
        comment!("Tile: Init Stage");

        RunningState::<AP::EA> {
            // TODO Neg infinity
            m: AP::EA::from_int(-99999999999),
            l: AP::EA::from_int(0),
        }
    }

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<AP::EO>>,
        #[comptime] tile_config: Self::Config,
    ) {
        FM::write_results(&acc.fragment, slice, tile_config)
    }
    fn tmp_write_score<G: GlobalAttentionConfig>(
        acc: &Self::ScoreProb,
        writer: &mut PlaneWriter<AP::EO>,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let mut out_smem =
            SharedMemory::<AP::EA>::new(tile_config.attention_tile_size().accumulator_size());

        FM::tmp_write_score::<AP::EA>(
            &acc.fragment,
            &mut out_smem.to_slice_mut().try_cast_unchecked(),
            tile_config,
        );

        PlaneWriter::<AP::EO>::write(
            writer,
            out_smem.to_slice().try_cast_unchecked(),
            0,
            0,
            1u32,
            tile_config.plane_dim(),
            global_config.global_memory_config(FlashIdent::Out),
        )
    }
    fn tmp_write_query<G: GlobalAttentionConfig>(
        query: &Self::Query,
        writer: &mut PlaneWriter<AP::EO>,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let mut out_smem =
            SharedMemory::<AP::EA>::new(tile_config.attention_tile_size().accumulator_size());

        FM::tmp_write_query::<AP::EA>(
            &query.fragment,
            &mut out_smem.to_slice_mut().try_cast_unchecked(),
            tile_config,
        );

        PlaneWriter::<AP::EO>::write(
            writer,
            out_smem.to_slice().try_cast_unchecked(),
            0,
            0,
            1u32,
            tile_config.plane_dim(),
            global_config.global_memory_config(FlashIdent::Out),
        )
    }
    fn tmp_write_key<G: GlobalAttentionConfig>(
        key: &Self::KeyValue,
        writer: &mut PlaneWriter<AP::EO>,
        #[comptime] tile_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let mut out_smem =
            SharedMemory::<AP::EA>::new(tile_config.attention_tile_size().accumulator_size());

        FM::tmp_write_key::<AP::EA>(
            &key.key(),
            &mut out_smem.to_slice_mut().try_cast_unchecked(),
            tile_config,
        );

        PlaneWriter::<AP::EO>::write(
            writer,
            out_smem.to_slice().try_cast_unchecked(),
            0,
            0,
            1u32,
            tile_config.plane_dim(),
            global_config.global_memory_config(FlashIdent::Out),
        )
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
        state: &Self::State,
        #[comptime] dk: u32,
    ) -> RowStats<AP::EA> {
        let inv_sqrt_dk = AP::EA::new(comptime!(1.0 / (dk as f32).sqrt()));

        score_prob.multiply_score(inv_sqrt_dk);

        match out_of_bound_mask {
            CubeOption::Some(out_of_bound_mask) => score_prob.apply_mask(out_of_bound_mask),
            CubeOption::None => {}
        }

        let m = score_prob.row_max(state.m);

        score_prob.to_prob(m);
        let prob_row_sum = score_prob.row_sum();

        RowStats::<AP::EA> { m, prob_row_sum }
    }

    fn accumulate_value(
        key_value: &Self::KeyValue,
        score_prob: &Self::ScoreProb,
        accumulator: &mut Self::Accumulator,
        score_prob_row_stats: &RowStats<AP::EA>,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    ) {
        let prev_m = state.m;
        let prev_l = state.l;
        let new_m = score_prob_row_stats.m;
        let row_sum = score_prob_row_stats.prob_row_sum;

        let exp_m_diff = Exp::exp(prev_m - new_m);
        let new_l = exp_m_diff * prev_l + row_sum;

        accumulator.scale(exp_m_diff);

        FM::value_matmul(
            &score_prob.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );

        state.m = new_m;
        state.l = new_l;
    }
}

#[derive(CubeType)]
pub struct RunningState<E: Float> {
    m: E,
    l: E,
}

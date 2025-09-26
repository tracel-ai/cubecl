use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use std::marker::PhantomData;

use crate::components::TileMask;
use crate::components::tile::AccumulatorTile as _;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::ScaleMode;
use crate::components::tile::SoftmaxTileExpand;
use crate::components::tile::dummy::DummyAccumulator;
use crate::components::tile::dummy::{DummySoftmax, FlashMatmul, FlashPrecision};
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

    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxTile {
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
        FM::zero_softmax(&mut score.fragment, config);
    }

    fn accumulate_score(
        query: &Self::QueryTile,
        key_value: &Self::KeyValueTile,
        softmax: &mut Self::SoftmaxTile,
        #[comptime] config: Self::Config,
    ) {
        FM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut softmax.fragment,
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

    fn accumulate_value(
        softmax: &Self::SoftmaxTile,
        key_value: &Self::KeyValueTile,
        accumulator: &mut Self::AccumulatorTile,
        scale: &RowWise<AP::EA>,
        #[comptime] config: Self::Config,
    ) {
        accumulator.scale(scale, ScaleMode::Multiply);

        FM::value_matmul(
            &softmax.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }
}

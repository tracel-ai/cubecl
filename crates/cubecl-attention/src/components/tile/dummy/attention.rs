use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use std::marker::PhantomData;

use crate::components::TileMask;
use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile as _;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::ScaleMode;
use crate::components::tile::SoftmaxTileExpand;
use crate::components::tile::dummy::DummyAccumulator;
use crate::components::tile::dummy::{AttentionMatmul, DummySoftmax};
use crate::components::tile::{RowWise, RunningState, SoftmaxTile, TileAttention};
use crate::components::{
    AttentionPrecision,
    tile::dummy::{KeyValueFragment, QueryFragment},
};

pub struct DummyTileAttention<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    _phantom: PhantomData<(AP, AM)>,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> TileAttention<AP>
    for DummyTileAttention<AP, AM>
{
    type Config = AM::Config;

    type QueryTile = QueryFragment<AP, AM>;
    type KeyValueTile = KeyValueFragment<AP, AM>;
    type SoftmaxTile = DummySoftmax<AP, AM>;
    type AccumulatorTile = DummyAccumulator<AP, AM>;

    fn rescale(
        acc: &mut Self::AccumulatorTile,
        prev_state: &RunningState<SM<AP>>,
        #[comptime] _config: Self::Config,
    ) {
        acc.scale(&prev_state.l.cast::<ACC<AP>>(), ScaleMode::Divide);
    }

    fn write_results(
        acc: &Self::AccumulatorTile,
        slice: &mut SliceMut<Line<OG<AP>>>,
        #[comptime] tile_config: Self::Config,
    ) {
        AM::write_results(&acc.fragment, slice, tile_config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorTile {
        Self::AccumulatorTile::new(config)
    }

    fn init_query(tile: &StridedTile<QG<AP>>, #[comptime] config: Self::Config) -> Self::QueryTile {
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
        AM::fill_key_value(tile, rhs.key_mut(), config);
    }

    fn fill_value<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    ) {
        AM::fill_key_value(tile, rhs.value_mut(), config);
    }

    fn zero_softmax(score: &mut Self::SoftmaxTile, #[comptime] config: Self::Config) {
        AM::zero_softmax(&mut score.fragment, config);
    }

    fn accumulate_score(
        query: &Self::QueryTile,
        key_value: &Self::KeyValueTile,
        softmax: &mut Self::SoftmaxTile,
        #[comptime] config: Self::Config,
    ) {
        AM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut softmax.fragment,
            config,
        );
    }

    fn softmax(
        softmax: &mut Self::SoftmaxTile,
        mask: TileMask,
        state: &mut RunningState<SM<AP>>,
        #[comptime] dk: u32,
    ) -> RowWise<ACC<AP>> {
        let inv_sqrt_dk = SM::<AP>::new(comptime!(1.0 / (dk as f32).sqrt()));

        softmax.scale_and_mask(inv_sqrt_dk, mask);

        let score_max = softmax.row_max(state.m.copy());

        softmax.to_prob(state, &score_max)
    }

    fn accumulate_value(
        softmax: &Self::SoftmaxTile,
        key_value: &Self::KeyValueTile,
        accumulator: &mut Self::AccumulatorTile,
        scale: &RowWise<ACC<AP>>,
        #[comptime] config: Self::Config,
    ) {
        accumulator.scale(scale, ScaleMode::Multiply);

        AM::value_matmul(
            &softmax.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::tile::AccumulatorTile as _;
use crate::components::tile::AccumulatorTileExpand;
use crate::components::tile::SoftmaxTileExpand;
use crate::components::tile::dummy::DummyAccumulator;
use crate::components::tile::dummy::MaskFragment;
use crate::components::tile::dummy::attention_matmul::AttentionMatmulConfig;
use crate::components::tile::dummy::{AttentionMatmul, DummySoftmax};
use crate::components::tile::tiles::{KeyValueTile, KeyValueTileExpand};
use crate::components::tile::tiles::{MaskTile, MaskTileExpand};
use crate::components::tile::tiles::{QueryTile, QueryTileExpand};
use crate::components::tile::{RowWise, RunningState, SoftmaxTile, TileAttention};
use crate::components::{
    AttentionPrecision,
    tile::dummy::{KeyValueFragment, QueryFragment},
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

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
    type MaskTile = MaskFragment<AP, AM>;

    fn rescale(
        acc: &mut Self::AccumulatorTile,
        prev_state: &RunningState<SM<AP>>,
        #[comptime] _config: Self::Config,
    ) {
        acc.scale_div(&prev_state.l);
    }

    fn write_results(
        tile: &mut StridedTile<OS<AP>, ReadWrite>,
        acc: &Self::AccumulatorTile,
        #[comptime] tile_config: Self::Config,
    ) {
        AM::write_results(&acc.fragment, &mut tile.slice, tile_config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorTile {
        Self::AccumulatorTile::new(config)
    }

    fn init_query(#[comptime] config: Self::Config) -> Self::QueryTile {
        Self::QueryTile::new(config)
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

    fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] partition_pos: Coords2d,
        #[comptime] config: Self::Config,
    ) -> Self::MaskTile {
        Self::MaskTile::new(out_of_bounds, partition_pos, config)
    }

    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxTile {
        Self::SoftmaxTile::new(config)
    }

    fn init_state(#[comptime] config: Self::Config) -> RunningState<SM<AP>> {
        RunningState::<SM<AP>>::init(config.num_rows_per_unit())
    }

    fn fill_query<E: Float>(tile: &StridedTile<E>, registers: &mut Self::QueryTile) {
        AM::fill_query(tile, registers.fragment_mut());
    }

    fn fill_key<E: Float>(
        tile: &StridedTile<E>,
        registers: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    ) {
        AM::fill_key_value(tile, registers.key_mut(), config);
    }

    fn fill_value<E: Float>(
        tile: &StridedTile<E>,
        registers: &mut Self::KeyValueTile,
        #[comptime] config: Self::Config,
    ) {
        AM::fill_key_value(tile, registers.value_mut(), config);
    }

    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::MaskTile,
        #[comptime] config: Self::Config,
    ) {
        AM::fill_mask(tile, mask.fragment_mut(), config)
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
        mask: &Self::MaskTile,
        state: &mut RunningState<SM<AP>>,
        max_placeholder: &mut RowWise<SM<AP>>,
        sum_placeholder: &mut RowWise<SM<AP>>,
        #[comptime] dk: u32,
        #[comptime] config: Self::Config,
    ) -> RowWise<SM<AP>> {
        Self::SoftmaxTile::scale_and_mask::<Self::MaskTile>(
            softmax,
            SM::<AP>::new(comptime!(1.0 / (dk as f32).sqrt())),
            mask,
        );

        softmax.row_max::<Self::Config>(max_placeholder, &state.m, config);

        softmax.to_prob::<Self::Config>(state, max_placeholder, sum_placeholder, config)
    }

    fn accumulate_value(
        softmax: &Self::SoftmaxTile,
        key_value: &Self::KeyValueTile,
        accumulator: &mut Self::AccumulatorTile,
        scale: &RowWise<SM<AP>>,
        #[comptime] config: Self::Config,
    ) {
        accumulator.scale_mul(scale);

        AM::value_matmul(
            &softmax.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }

    fn init_max_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>> {
        RowWise::new_min_value(num_rows)
    }

    fn init_sum_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>> {
        RowWise::new_zero(num_rows)
    }
}

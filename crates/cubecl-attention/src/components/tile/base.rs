use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
    tile::StridedTile,
};

use crate::components::tile::{AccumulatorTile, KeyValueTile, MaskTile, QueryTile, SoftmaxTile};
use crate::components::{
    AttentionPrecision,
    attention_types::*,
    fragment::AttentionMatmulConfig,
    tile::{RowWise, RunningState},
};
use std::marker::PhantomData;

use crate::components::fragment::AttentionMatmul;
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

#[derive(CubeType)]
pub struct TileAttention<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    #[cube(comptime)]
    _phantom: PhantomData<(AP, AM)>,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> TileAttention<AP, AM> {
    pub fn rescale(acc: &mut AccumulatorTile<AP, AM>, prev_state: &RunningState<SM<AP>>) {
        acc.scale_div(prev_state.l());
    }

    pub fn write_results(
        tile: &mut StridedTile<OS<AP>, ReadWrite>,
        acc: &AccumulatorTile<AP, AM>,
        #[comptime] config: AM::Config,
    ) {
        AM::write_results(&acc.fragment, &mut tile.slice, config)
    }

    pub fn init_accumulator(#[comptime] config: AM::Config) -> AccumulatorTile<AP, AM> {
        AccumulatorTile::new(config)
    }

    pub fn init_query(#[comptime] config: AM::Config) -> QueryTile<AP, AM> {
        QueryTile::new(config)
    }

    pub fn init_key_value(#[comptime] config: AM::Config) -> KeyValueTile<AP, AM> {
        KeyValueTile::new_key_value(config)
    }

    pub fn init_key(#[comptime] config: AM::Config) -> KeyValueTile<AP, AM> {
        KeyValueTile::new_key(config)
    }

    pub fn init_value(#[comptime] config: AM::Config) -> KeyValueTile<AP, AM> {
        KeyValueTile::new_value(config)
    }

    pub fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] partition_pos: Coords2d,
        #[comptime] config: AM::Config,
    ) -> MaskTile<AP, AM> {
        MaskTile::new(out_of_bounds, partition_pos, config)
    }

    pub fn init_softmax(#[comptime] config: AM::Config) -> SoftmaxTile<AP, AM> {
        SoftmaxTile::new(config)
    }

    pub fn init_state(#[comptime] config: AM::Config) -> RunningState<SM<AP>> {
        RunningState::<SM<AP>>::init(config.num_rows_per_unit())
    }

    pub fn fill_key<E: Float>(
        tile: &StridedTile<E>,
        registers: &mut KeyValueTile<AP, AM>,
        #[comptime] config: AM::Config,
    ) {
        AM::fill_key_value(tile, registers.key_mut(), config);
    }

    pub fn fill_value<E: Float>(
        tile: &StridedTile<E>,
        registers: &mut KeyValueTile<AP, AM>,
        #[comptime] config: AM::Config,
    ) {
        AM::fill_key_value(tile, registers.value_mut(), config);
    }

    pub fn zero_softmax(score: &mut SoftmaxTile<AP, AM>, #[comptime] config: AM::Config) {
        AM::zero_softmax(&mut score.fragment, config);
    }

    pub fn accumulate_score(
        query: &QueryTile<AP, AM>,
        key_value: &KeyValueTile<AP, AM>,
        softmax: &mut SoftmaxTile<AP, AM>,
        #[comptime] config: AM::Config,
    ) {
        AM::score_matmul(
            &query.fragment,
            key_value.key(),
            &mut softmax.fragment,
            config,
        );
    }

    pub fn softmax(
        softmax: &mut SoftmaxTile<AP, AM>,
        mask: &MaskTile<AP, AM>,
        state: &mut RunningState<SM<AP>>,
        max_placeholder: &mut RowWise<SM<AP>>,
        sum_placeholder: &mut RowWise<SM<AP>>,
        #[comptime] dk: u32,
        #[comptime] config: AM::Config,
    ) -> RowWise<SM<AP>> {
        SoftmaxTile::scale_and_mask(
            softmax,
            SM::<AP>::new(comptime!(1.0 / (dk as f32).sqrt())),
            mask,
        );

        softmax.row_max::<AM::Config>(max_placeholder, state.m(), config);

        softmax.to_prob::<AM::Config>(state, max_placeholder, sum_placeholder, config)
    }

    pub fn accumulate_value(
        softmax: &SoftmaxTile<AP, AM>,
        key_value: &KeyValueTile<AP, AM>,
        accumulator: &mut AccumulatorTile<AP, AM>,
        scale: &RowWise<SM<AP>>,
        #[comptime] config: AM::Config,
    ) {
        accumulator.scale_mul(scale);

        AM::value_matmul(
            &softmax.fragment,
            key_value.value(),
            &mut accumulator.fragment,
            config,
        );
    }

    pub fn init_max_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>> {
        RowWise::new_min_value(num_rows)
    }

    pub fn init_sum_placeholder(#[comptime] num_rows: u32) -> RowWise<SM<AP>> {
        RowWise::new_zero(num_rows)
    }
}

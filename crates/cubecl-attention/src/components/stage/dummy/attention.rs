use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::StageToTileReader;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords3d;
use std::marker::PhantomData;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::stage::dummy::DummyStageConfig;
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};

pub struct DummyStageAttention<AP: AttentionPrecision, R, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, R, TA)>,
}

#[cube]
impl<AP: AttentionPrecision, R: StageToTileReader<AP::ES>, TA: TileAttention<AP>> StageAttention<AP>
    for DummyStageAttention<AP, R, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyReader = R;
    type ValueReader = R;

    type State = TA::State;
    type Query = TA::Query;
    type KeyValue = TA::KeyValue;
    type Score = TA::Score;
    type Accumulator = TA::Accumulator;
    type Writer = TA::Writer;

    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        #[comptime] config: Self::Config,
    ) {
        let key_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
            <Self::Config as StageAttentionConfig>::ScoreStageMemoryConfig,
        >(key_reader, 0, 0, config.score_stage_memory_config());
        let value_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
            <Self::Config as StageAttentionConfig>::ValueStageMemoryConfig,
        >(value_reader, 0, 0, config.value_stage_memory_config());

        TA::execute(
            &key_tile,
            &value_tile,
            query,
            key_value,
            score_prob,
            accumulator,
            state,
            config.tile_config(),
        );
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] config: Self::Config) {
        comment!("Stage: Rescale");
        TA::rescale(acc, state, config.tile_config())
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::State {
        comment!("Stage: Init Stage");
        TA::init_state(config.tile_config())
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        comment!("Stage: Write");
        TA::write::<G>(acc, writer, stage_config.tile_config(), global_config);
    }

    fn init_writer(out: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer {
        TA::init_writer(out)
    }

    fn init_fragments(
        query_reader: QueryRegisterReader<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> (Self::Query, Self::KeyValue, Self::Score, Self::Accumulator) {
        TA::init_fragments(query_reader, config.tile_config())
    }
}

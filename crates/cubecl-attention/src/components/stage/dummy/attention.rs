use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::StageToTileReader;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::global::dummy::QueryRegisterReader;
use crate::components::stage::dummy::DummyStageConfig;
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{
    AttentionPrecision, global::GlobalAttentionConfig, tile::dummy::DummyWriter,
};

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
    type Writer = DummyWriter<AP::EO>;

    type State = DummyStageState<AP::EA>;

    type Query = TA::Query;
    type KeyValue = TA::KeyValue;
    type ScoreProb = TA::ScoreProb;
    type Accumulator = TA::Accumulator;

    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::ScoreProb,
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
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] config: Self::Config) {
        comment!("Stage: Rescale");
    }

    fn init_state(#[comptime] _config: Self::Config) -> Self::State {
        comment!("Stage: Init Stage");

        DummyStageState::<AP::EA> {
            // TODO Neg infinity
            m: AP::EA::from_int(-99999999999),
            l: AP::EA::from_int(0),
        }
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        comment!("Stage: Write");
    }

    fn init_writer(out: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer {
        DummyWriter::new(out, 0, 0, 0)
    }

    fn init_fragments(
        query_reader: QueryRegisterReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (
        Self::Query,
        Self::KeyValue,
        Self::ScoreProb,
        Self::Accumulator,
    ) {
        TA::init_fragments(query_reader, config.tile_config())
    }
}

#[derive(CubeType)]
// There should be two strategies for state
// - Elect: one thread holds the state and shares it with row neighbours when necessary (needs broadcast at the beginning)
// - Duplicate: all neighbours hold the value (needs broadcast at the end)
//
// Note: this assumes plane_dim >= row count and plane_dim % row count == 0
pub struct DummyStageState<E: Float> {
    // Equal m_i'(j-1)
    m: E,
    l: E,
}

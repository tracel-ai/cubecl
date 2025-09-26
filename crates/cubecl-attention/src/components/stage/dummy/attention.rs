use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::stage::dummy::StageState;
use crate::components::stage::dummy::{Accumulators, DummyStageConfig, KeyValues, Queries, Scores};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{StageMask, global::dummy::QueryReader};

pub struct DummyStageAttention<AP: AttentionPrecision, S, SO, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, S, SO, TA)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    S: Stage<AP::ES, ReadOnly, TileKind = Strided>,
    SO: Stage<AP::EO, ReadWrite, TileKind = Strided>,
    TA: TileAttention<AP>,
> StageAttention<AP> for DummyStageAttention<AP, S, SO, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyStage = S;
    type ValueStage = S;
    type OutStage = SO;

    type State = StageState<AP>;
    type Query = Queries<AP, TA, Self::Config>;
    type KeyValue = KeyValues<AP, TA, Self::Config>;
    type Score = Scores<AP, TA, Self::Config>;
    type Accumulator = Accumulators<AP, TA, Self::Config>;

    fn execute(
        key_reader: &Self::KeyStage,
        value_reader: &Self::ValueStage,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::Score,
        mask: StageMask,
        accumulator: &mut Self::Accumulator,
        state: &mut StageState<AP>,
        #[comptime] config: Self::Config,
    ) {
        let partition_mask = mask.to_partition(UNIT_POS_Y);

        let p = config.tiling_scheme().partition_size;

        let mut kv = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_kv {
            let mut hd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.head_dim {
                let key_tile = S::tile(key_reader, (hd, kv).runtime());

                TA::fill_key(
                    &key_tile,
                    key_value.get_key_at_mut(hd, kv, config),
                    config.tile_config(),
                );

                comptime![hd += 1];
            }

            let mut q = comptime![0u32];
            let mut scales = Sequence::<AP::EA>::new();

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.seq_q {
                let score_frag = score_prob.get_at_mut(q, kv, config);
                TA::zero_score(score_frag, config.tile_config());

                let mut hd = comptime![0u32];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.head_dim {
                    let query_frag = query.get_at(q, hd, config);
                    let key_frag = key_value.get_key_at(hd, kv, config);

                    TA::accumulate_score(query_frag, key_frag, score_frag, config.tile_config());

                    comptime![hd += 1];
                }

                let state_q = state.get_at_mut(q);

                let row_stats = TA::score_to_prob(
                    score_frag,
                    partition_mask.to_tile(q, kv),
                    state_q,
                    config.tiling_scheme().elements_in_partition_head_dim(),
                );

                scales.push(TA::update_state(state_q, &row_stats));

                comptime![q += 1];
            }

            let mut vd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                let value_tile = S::tile(value_reader, (kv, vd).runtime());

                TA::fill_value(
                    &value_tile,
                    key_value.get_value_at_mut(kv, vd, config),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            let mut q = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.seq_q {
                let mut vd = comptime![0u32];
                let score_frag = score_prob.get_at(q, kv, config);

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.val_dim {
                    TA::accumulate_value(
                        score_frag,
                        key_value.get_value_at(kv, vd, config),
                        accumulator.get_at_mut(q, vd, config),
                        *scales.index(q),
                        config.tile_config(),
                    );

                    comptime![vd += 1];
                }

                comptime![q += 1];
            }

            comptime![kv += 1];
        }
    }

    fn rescale(
        acc: &mut Self::Accumulator,
        state: StageState<AP>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut q = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut vd = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                TA::rescale(
                    Self::Accumulator::get_at_mut(acc, q, vd, config),
                    state.get_at(q),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            comptime![q += 1];
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> StageState<AP> {
        StageState::<AP>::init::<Self::Config>(config)
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        stage: &mut Self::OutStage,
        writer: &mut W,
        #[comptime] stage_config: Self::Config,
    ) {
        let p = stage_config.tiling_scheme().partition_size;
        let mut q = comptime!(0u32);

        W::on_event(writer, WriteEvent::new_Begin());

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut kv = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                let tile_pos = (q + UNIT_POS_Y * p.seq_q, kv.runtime());
                let mut tile = Self::OutStage::tile(stage, tile_pos);

                TA::write_results(
                    &mut tile,
                    Self::Accumulator::get_at(acc, q, kv, stage_config),
                    stage_config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));

                comptime![kv += 1];
            }

            comptime![q += 1];
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_fragments(
        query_loader: QueryReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (Self::Query, Self::KeyValue, Self::Score, Self::Accumulator) {
        (
            Self::Query::new(query_loader, config),
            Self::KeyValue::new(config),
            Self::Score::new(config),
            Self::Accumulator::new(config),
        )
    }
}

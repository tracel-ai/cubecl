use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::StageToTileReader;
use cubecl_std::CubeOption;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords3d;
use std::marker::PhantomData;

use crate::components::global::dummy::QueryLoader;
use crate::components::stage::dummy::{
    Accumulators, AttentionStageMemoryConfig, DummyStageConfig, KeyValues, Queries, Scores,
};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::RowStats;
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
    type Query = Queries<AP, TA, Self::Config>;
    type KeyValue = KeyValues<AP, TA, Self::Config>;
    type Score = Scores<AP, TA, Self::Config>;
    type Accumulator = Accumulators<AP, TA, Self::Config>;
    type Writer = TA::Writer;

    // for kv in 0..seq_kv:
    //     load key[:, kv] into KeyValue
    //     for q in 0..seq_q:
    //         score[q] = zeros()
    //         for hd in 0..head_dim:
    //              score[q] += query[q,hd] · KeyValue[hd]
    //     load val[kv,:] into KeyValue
    //     for q in 0..seq_q:
    //         for vd in 0..val_dim:
    //             acc[q,vd] += score[q] * KeyValue[vd]
    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        state: &mut Self::State,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        #[comptime] config: Self::Config,
    ) {
        let (seq_q, head_dim, seq_kv, val_dim) = comptime! {let p = config.tiling_scheme().partition_size; (p.seq_q, p.head_dim, p.seq_kv, p.val_dim)};

        let mut kv = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..seq_kv {
            let mut hd = comptime![0u32];

            // TODO: if seq_q=1 skip preloading, do on the fly
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..head_dim {
                let key_tile =
                    <R as StageToTileReader<AP::ES>>::read_tile::<AttentionStageMemoryConfig>(
                        key_reader,
                        // TODO maybe transpose the indexes?
                        kv,
                        hd,
                        config.score_stage_memory_config(),
                    );

                TA::fill_key(
                    &key_tile,
                    key_value.get_at_mut(hd, config),
                    config.tile_config(),
                );

                comptime![hd += 1];
            }

            let mut q = comptime![0u32];

            let mut row_stats: Sequence<RowStats<AP::EA>> = Sequence::new();

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..seq_q {
                let score_frag = score_prob.get_at_mut(q);
                TA::zero_score(score_frag, config.tile_config());

                let mut hd = comptime![0u32];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..head_dim {
                    let query_frag = query.get_at(q, hd, config);
                    let key_frag = key_value.get_at(hd, config);

                    TA::accumulate_score(query_frag, key_frag, score_frag, config.tile_config());

                    comptime![hd += 1];
                }

                row_stats.push(TA::score_to_prob(
                    score_frag,
                    out_of_bound_mask,
                    state,
                    config.tile_config(),
                ));

                comptime![q += 1];
            }

            let mut vd = comptime![0u32];

            // TODO: if seq_q=1 skip preloading, do on the fly
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..val_dim {
                let value_tile = <R as StageToTileReader<AP::ES>>::read_tile::<
                    AttentionStageMemoryConfig,
                >(
                    value_reader, kv, vd, config.value_stage_memory_config()
                );

                TA::fill_value(
                    &value_tile,
                    key_value.get_at_mut(vd, config),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            let mut q = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..seq_q {
                let mut vd = comptime![0u32];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..val_dim {
                    TA::accumulate_value(
                        key_value.get_at(vd, config),
                        score_prob.get_at(q),
                        accumulator.get_at_mut(q, vd, config),
                        row_stats.index(q),
                        state,
                        config.tile_config(),
                    );

                    comptime![vd += 1];
                }

                comptime![q += 1];
            }

            comptime![kv += 1];
        }
    }

    fn rescale(acc: &mut Self::Accumulator, state: Self::State, #[comptime] config: Self::Config) {
        comment!("Stage: Rescale");
        let partition_size = config.tiling_scheme().partition_size;

        let mut i = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime!(partition_size.seq_q) {
            let mut j = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime!(partition_size.val_dim) {
                TA::rescale(
                    Self::Accumulator::get_at_mut(acc, i, j, config),
                    &state,
                    config.tile_config(),
                );

                comptime![j += 1];
            }

            comptime![i += 1];
        }
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
        let partition_size = stage_config.tiling_scheme().partition_size;

        let mut i = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..comptime!(partition_size.seq_q) {
            let mut j = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..comptime!(partition_size.val_dim) {
                TA::write::<G>(
                    Self::Accumulator::get_at(acc, i, j, stage_config),
                    writer,
                    stage_config.tile_config(),
                    global_config,
                );

                comptime![j += 1];
            }

            comptime![i += 1];
        }
    }

    fn init_writer(q_offset: u32, out: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer {
        TA::init_writer(q_offset, out)
    }

    fn init_fragments(
        query_loader: QueryLoader<AP>,
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

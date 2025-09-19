use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::PlaneWriter;
use cubecl_matmul::components::global::StageUnloader as _;
use cubecl_matmul::components::stage::StageReader;
use cubecl_matmul::components::tile::loader::Strided;
use cubecl_std::CubeOption;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords3d;
use std::marker::PhantomData;

use crate::components::FlashIdent;
use crate::components::global::dummy::QueryLoader;
use crate::components::stage::dummy::StageState;
use crate::components::stage::dummy::{
    Accumulators, AttentionStageMemoryConfig, DummyStageConfig, KeyValues, Queries, Scores,
};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::tile::dummy::RunningState;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};

pub struct DummyStageAttention<AP: AttentionPrecision, R, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, R, TA)>,
}

#[cube]
impl<AP: AttentionPrecision, R: StageReader<AP::ES, TileKind = Strided>, TA: TileAttention<AP>>
    StageAttention<AP> for DummyStageAttention<AP, R, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyReader = R;
    type ValueReader = R;

    type State = StageState<AP>;
    type Query = Queries<AP, TA, Self::Config>;
    type KeyValue = KeyValues<AP, TA, Self::Config>;
    type Score = Scores<AP, TA, Self::Config>;
    type Accumulator = Accumulators<AP, TA, Self::Config>;
    type Writer = PlaneWriter<AP::EO>;

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
    // fn execute(
    //     key_reader: &Self::KeyReader,
    //     value_reader: &Self::ValueReader,
    //     query: &Self::Query,
    //     key_value: &mut Self::KeyValue,
    //     score_prob: &mut Self::Score,
    //     accumulator: &mut Self::Accumulator,
    //     state: &mut Self::State,
    //     out_of_bound_mask: CubeOption<(u32, u32)>,
    //     #[comptime] config: Self::Config,
    // ) {
    //     let p = config.tiling_scheme().partition_size;

    //     let mut kv = comptime![0u32];

    //     #[unroll]
    //     #[allow(clippy::explicit_counter_loop)]
    //     for _ in 0..p.seq_kv {
    //         let mut hd = comptime![0u32];

    //         // TODO: if p.seq_q=1 skip preloading, do on the fly
    //         #[unroll]
    //         #[allow(clippy::explicit_counter_loop)]
    //         for _ in 0..p.head_dim {
    //             let key_tile = <R as StageReader<AP::ES>>::read_tile::<AttentionStageMemoryConfig>(
    //                 key_reader,
    //                 hd,
    //                 kv,
    //                 config.score_stage_memory_config(),
    //             );

    //             TA::fill_key(
    //                 &key_tile,
    //                 key_value.get_key_at_mut(hd, kv, config),
    //                 config.tile_config(),
    //             );

    //             comptime![hd += 1];
    //         }

    //         let mut q = comptime![0u32];

    //         let mut row_stats: Sequence<RowStats<AP::EA>> = Sequence::new();

    //         #[unroll]
    //         #[allow(clippy::explicit_counter_loop)]
    //         for _ in 0..p.seq_q {
    //             let score_frag = score_prob.get_at_mut(q, kv, config);
    //             TA::zero_score(score_frag, config.tile_config());

    //             let mut hd = comptime![0u32];

    //             #[unroll]
    //             #[allow(clippy::explicit_counter_loop)]
    //             for _ in 0..p.head_dim {
    //                 let query_frag = query.get_at(q, hd, config);
    //                 let key_frag = key_value.get_key_at(hd, kv, config);

    //                 TA::accumulate_score(query_frag, key_frag, score_frag, config.tile_config());

    //                 comptime![hd += 1];
    //             }

    //             row_stats.push(TA::score_to_prob(
    //                 score_frag,
    //                 out_of_bound_mask,
    //                 state.get_at(q),
    //                 config.tiling_scheme().head_dim(),
    //             ));

    //             comptime![q += 1];
    //         }

    //         let mut vd = comptime![0u32];

    //         // TODO: if p.seq_q=1 skip preloading, do on the fly
    //         #[unroll]
    //         #[allow(clippy::explicit_counter_loop)]
    //         for _ in 0..p.val_dim {
    //             let value_tile = <R as StageReader<AP::ES>>::read_tile::<AttentionStageMemoryConfig>(
    //                 value_reader,
    //                 kv,
    //                 vd,
    //                 config.value_stage_memory_config(),
    //             );

    //             TA::fill_value(
    //                 &value_tile,
    //                 key_value.get_value_at_mut(kv, vd, config),
    //                 config.tile_config(),
    //             );

    //             comptime![vd += 1];
    //         }

    //         let mut q = comptime![0u32];

    //         #[unroll]
    //         #[allow(clippy::explicit_counter_loop)]
    //         for _ in 0..p.seq_q {
    //             let mut vd = comptime![0u32];

    //             let scale = TA::update_state(row_stats.index(q), state.get_at_mut(q));

    //             #[unroll]
    //             #[allow(clippy::explicit_counter_loop)]
    //             for _ in 0..p.val_dim {
    //                 TA::accumulate_value(
    //                     key_value.get_value_at(kv, vd, config),
    //                     score_prob.get_at(q, kv, config),
    //                     accumulator.get_at_mut(q, vd, config),
    //                     scale,
    //                     config.tile_config(),
    //                 );

    //                 comptime![vd += 1];
    //             }

    //             comptime![q += 1];
    //         }

    //         comptime![kv += 1];
    //     }
    // }

    fn execute(
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        query: &Self::Query,
        key_value: &mut Self::KeyValue,
        score_prob: &mut Self::Score,
        accumulator: &mut Self::Accumulator,
        state: &mut StageState<AP>,
        out_of_bound_mask: CubeOption<(u32, u32)>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut kv = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_kv {
            let mut hd = comptime![0u32];

            // TODO: if p.seq_q=1 skip preloading, do on the fly
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.head_dim {
                let key_tile = <R as StageReader<AP::ES>>::read_tile::<AttentionStageMemoryConfig>(
                    key_reader,
                    hd,
                    kv,
                    config.score_stage_memory_config(),
                );

                TA::fill_key(
                    &key_tile,
                    key_value.get_key_at_mut(hd, kv, config),
                    config.tile_config(),
                );

                comptime![hd += 1];
            }

            let mut vd = comptime![0u32];

            // TODO: if p.seq_q=1 skip preloading, do on the fly
            // TODO: move to later if reuse key
            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                let value_tile = <R as StageReader<AP::ES>>::read_tile::<AttentionStageMemoryConfig>(
                    value_reader,
                    kv,
                    vd,
                    config.value_stage_memory_config(),
                );

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
                    out_of_bound_mask,
                    state_q.m,
                    config.tiling_scheme().head_dim(),
                );

                let (exp_m_diff, m_new, l_new) =
                    TA::get_new_state(&row_stats, state_q.m, state_q.l);

                let mut vd = comptime![0u32];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.val_dim {
                    TA::accumulate_value(
                        score_frag,
                        key_value.get_value_at(kv, vd, config),
                        accumulator.get_at_mut(q, vd, config),
                        exp_m_diff,
                        config.tile_config(),
                    );

                    comptime![vd += 1];
                }

                state_q.update(m_new, l_new);

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
        comment!("Stage: Rescale");
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
                    &state.get_at(q),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            comptime![q += 1];
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> StageState<AP> {
        comment!("Stage: Init Stage");
        StageState::<AP>::init::<Self::Config>(config)
    }

    fn init_writer(q_offset: u32, tensor: View<Line<AP::EO>, Coords3d, ReadWrite>) -> Self::Writer {
        PlaneWriter::new(tensor, q_offset, 0u32, 0u32)
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        comment!("Stage: Write");
        let p = stage_config.tiling_scheme().partition_size;
        let t = stage_config.tiling_scheme().tile_size;

        let out_smem_num_elements = p.seq_q * t.seq_q * p.val_dim * t.val_dim;

        let mut out_smem = SharedMemory::<AP::EO>::new_lined(out_smem_num_elements, 1u32);
        // TODO change indexes when we have planes>1
        let mut smem_slice = out_smem.slice_mut(0u32, out_smem_num_elements);

        let mut q = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut kv = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                TA::write_results(
                    Self::Accumulator::get_at(acc, q, kv, stage_config),
                    &mut smem_slice,
                    stage_config.tile_config(),
                );

                Self::Writer::write(
                    writer,
                    smem_slice.to_slice(),
                    q,
                    kv,
                    1u32,
                    stage_config.plane_dim(),
                    global_config.global_memory_config(FlashIdent::Out),
                );

                comptime![kv += 1];
            }

            comptime![q += 1];
        }
    }

    fn tmp_write_score<G: GlobalAttentionConfig>(
        score: &Self::Score,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let q = 0u32;
        let kv = 0u32;

        TA::tmp_write_score::<G>(
            Self::Score::get_at(score, q, kv, stage_config),
            writer,
            stage_config.tile_config(),
            global_config,
        );
    }
    fn tmp_write_query<G: GlobalAttentionConfig>(
        query: &Self::Query,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let q = 0u32;
        let hd = 0u32;

        TA::tmp_write_query::<G>(
            Self::Query::get_at(query, q, hd, stage_config),
            writer,
            stage_config.tile_config(),
            global_config,
        );
    }
    fn tmp_write_key<G: GlobalAttentionConfig>(
        key: &Self::KeyValue,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let hd = 0u32;
        let kv = 0u32;

        TA::tmp_write_key::<G>(
            Self::KeyValue::get_key_at(key, hd, kv, stage_config),
            writer,
            stage_config.tile_config(),
            global_config,
        );
    }

    fn tmp_write_value<G: GlobalAttentionConfig>(
        value: &Self::KeyValue,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let kv = 1u32;
        let vd = 0u32;

        TA::tmp_write_value::<G>(
            Self::KeyValue::get_value_at(value, kv, vd, stage_config),
            writer,
            stage_config.tile_config(),
            global_config,
        );
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

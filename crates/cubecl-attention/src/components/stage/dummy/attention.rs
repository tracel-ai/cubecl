use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    MatrixLayout,
    global::{GlobalWriter, PlaneWriter, memory::GlobalMemoryConfig},
    stage::Stage,
    tile::{StridedTile, io::Strided},
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords2d;
use std::marker::PhantomData;

use crate::components::stage::dummy::StageState;
use crate::components::stage::dummy::{Accumulators, DummyStageConfig, KeyValues, Queries, Scores};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{FlashIdent, global::dummy::QueryReader};

pub struct DummyStageAttention<AP: AttentionPrecision, R, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, R, TA)>,
}

#[cube]
impl<AP: AttentionPrecision, S: Stage<AP::ES, ReadOnly, TileKind = Strided>, TA: TileAttention<AP>>
    StageAttention<AP> for DummyStageAttention<AP, S, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyStage = S;
    type ValueStage = S;

    type State = StageState<AP>;
    type Query = Queries<AP, TA, Self::Config>;
    type KeyValue = KeyValues<AP, TA, Self::Config>;
    type Score = Scores<AP, TA, Self::Config>;
    type Accumulator = Accumulators<AP, TA, Self::Config>;
    type Writer = PlaneWriter<(AP::EO, AP::EO)>;

    fn execute(
        key_reader: &Self::KeyStage,
        value_reader: &Self::ValueStage,
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

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.head_dim {
                let key_tile = S::read_tile(key_reader, (hd, kv).runtime());

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
                    out_of_bound_mask,
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
                let value_tile = S::read_tile(value_reader, (kv, vd).runtime());

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

    fn init_writer(
        tensor: View<Line<AP::EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer {
        PlaneWriter::new(tensor, config, stage_config)
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::Accumulator,
        writer: &mut Self::Writer,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let p = stage_config.tiling_scheme().partition_size;

        let out_smem_num_elements = stage_config.tiling_scheme().elements_in_partition_seq_q()
            * stage_config.tiling_scheme().elements_in_partition_val_dim();

        let mut out_smem = SharedMemory::<AP::EO>::new_lined(
            comptime!(out_smem_num_elements * stage_config.num_planes()),
            1u32,
        );

        let start = UNIT_POS_Y * out_smem_num_elements;
        let end = start + out_smem_num_elements;
        let mut smem_slice = out_smem.slice_mut(start, end);

        let tile = StridedTile::new_strided_mut(
            smem_slice,
            stage_config.tiling_scheme().elements_in_partition_seq_q(),
            MatrixLayout::RowMajor,
        );

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
                    &tile,
                    (q + UNIT_POS_Y * p.seq_q, kv.runtime()),
                    stage_config.plane_dim(),
                    global_config.global_memory_config(FlashIdent::Out),
                );

                comptime![kv += 1];
            }

            comptime![q += 1];
        }
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

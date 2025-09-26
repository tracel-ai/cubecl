use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{GlobalWriter, PlaneWriter, memory::GlobalMemoryConfig},
    stage::Stage,
    tile::reader::Strided,
};
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords2d;
use std::marker::PhantomData;

use crate::components::StageMask;
use crate::components::stage::dummy::{
    Accumulators, DummyStageConfig, KeyValues, Queries, SoftmaxPartition,
};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{FlashIdent, global::dummy::QueryReader};
use crate::components::{stage::dummy::StageState, tile::RowWise};

pub struct DummyStageAttention<AP: AttentionPrecision, R, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, R, TA)>,
}

#[cube]
impl<AP: AttentionPrecision, S: Stage<AP::ES, TileKind = Strided>, TA: TileAttention<AP>>
    StageAttention<AP> for DummyStageAttention<AP, S, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyStage = S;
    type ValueStage = S;

    type State = StageState<AP>;
    type QueryPartition = Queries<AP, TA, Self::Config>;
    type KeyValuePartition = KeyValues<AP, TA, Self::Config>;
    type SoftmaxPartition = SoftmaxPartition<AP, TA, Self::Config>;
    type AccumulatorPartition = Accumulators<AP, TA, Self::Config>;
    type Writer = PlaneWriter<AP::EO>;

    fn execute(
        key_reader: &Self::KeyStage,
        value_reader: &Self::ValueStage,
        query_partition: &Self::QueryPartition,
        key_value_partition: &mut Self::KeyValuePartition,
        softmax_partition: &mut Self::SoftmaxPartition,
        mask: StageMask,
        accumulator_partition: &mut Self::AccumulatorPartition,
        state: &mut Self::State,
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
                let key_smem_slice =
                    <S as Stage<AP::ES>>::read_tile(key_reader, (hd, kv).runtime());

                TA::fill_key(
                    &key_smem_slice,
                    key_value_partition.get_key_at_mut(hd, kv, config),
                    config.tile_config(),
                );

                comptime![hd += 1];
            }

            let mut q = comptime![0u32];
            let mut scales = Sequence::<RowWise<AP::EA>>::new();

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.seq_q {
                let softmax_tile = softmax_partition.get_at_mut(q, kv, config);
                TA::zero_softmax(softmax_tile, config.tile_config());

                let mut hd = comptime![0u32];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.head_dim {
                    let query_tile = query_partition.get_at(q, hd, config);
                    let key_tile = key_value_partition.get_key_at(hd, kv, config);

                    TA::accumulate_score(query_tile, key_tile, softmax_tile, config.tile_config());

                    comptime![hd += 1];
                }

                let state_q = state.get_at_mut(q);

                let accumulator_scale = TA::softmax(
                    softmax_tile,
                    partition_mask.to_tile(q, kv),
                    state_q,
                    config.tiling_scheme().elements_in_partition_head_dim(),
                );

                scales.push(accumulator_scale);

                // let row_stats = TA::score_to_prob(
                //     softmax,
                //     partition_mask.to_tile(q, kv),
                //     state_q,
                //     config.tiling_scheme().elements_in_partition_head_dim(),
                // );

                // TA::update_state(state_q, &row_stats));

                comptime![q += 1];
            }

            let mut vd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                let value_smem_slice =
                    <S as Stage<AP::ES>>::read_tile(value_reader, (kv, vd).runtime());

                TA::fill_value(
                    &value_smem_slice,
                    key_value_partition.get_value_at_mut(kv, vd, config),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            let mut q = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.seq_q {
                let mut vd = comptime![0u32];
                let score_frag = softmax_partition.get_at(q, kv, config);

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.val_dim {
                    TA::accumulate_value(
                        score_frag,
                        key_value_partition.get_value_at(kv, vd, config),
                        accumulator_partition.get_at_mut(q, vd, config),
                        scales.index(q),
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
        acc: &mut Self::AccumulatorPartition,
        state: Self::State,
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
                    Self::AccumulatorPartition::get_at_mut(acc, q, vd, config),
                    state.get_at(q),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            comptime![q += 1];
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::State {
        StageState::<AP>::init::<Self::Config>(config)
    }

    fn init_writer(
        tensor: View<Line<AP::EO>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
    ) -> Self::Writer {
        PlaneWriter::new(tensor, config)
    }

    fn write<G: GlobalAttentionConfig>(
        acc: &Self::AccumulatorPartition,
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

        let mut q = comptime!(0u32);

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut kv = comptime!(0u32);

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                TA::write_results(
                    Self::AccumulatorPartition::get_at(acc, q, kv, stage_config),
                    &mut smem_slice,
                    stage_config.tile_config(),
                );

                Self::Writer::write(
                    writer,
                    smem_slice.to_slice(),
                    (q + UNIT_POS_Y * p.seq_q, kv.runtime()),
                    stage_config.plane_dim(),
                    global_config.global_memory_config(FlashIdent::Out),
                );

                comptime![kv += 1];
            }

            comptime![q += 1];
        }
    }

    fn init_partitions(
        query_loader: QueryReader<AP>,
        #[comptime] config: Self::Config,
    ) -> (
        Self::QueryPartition,
        Self::KeyValuePartition,
        Self::SoftmaxPartition,
        Self::AccumulatorPartition,
    ) {
        (
            Self::QueryPartition::new(query_loader, config),
            Self::KeyValuePartition::new(config),
            Self::SoftmaxPartition::new(config),
            Self::AccumulatorPartition::new(config),
        )
    }
}

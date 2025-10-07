use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::global::dummy::QueryReader;
use crate::components::stage::dummy::SoftmaxPartition;
use crate::components::stage::dummy::{Accumulators, DummyStageConfig, KeyValues, Queries};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::RowWise;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{StageMask, tile::RunningState};

pub struct DummyStageAttention<AP: AttentionPrecision, SK, SV, SO, TA: TileAttention<AP>> {
    _phantom: PhantomData<(AP, SK, SV, SO, TA)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, ReadOnly, TileKind = Strided>,
    SV: Stage<VS<AP>, ReadOnly, TileKind = Strided>,
    SO: Stage<OS<AP>, ReadWrite, TileKind = Strided>,
    TA: TileAttention<AP>,
> StageAttention<AP> for DummyStageAttention<AP, SK, SV, SO, TA>
{
    type Config = DummyStageConfig<TA::Config>;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type QueryPartition = Queries<AP, TA, Self::Config>;
    type KeyValuePartition = KeyValues<AP, TA, Self::Config>;
    type SoftmaxPartition = SoftmaxPartition<AP, TA, Self::Config>;
    type AccumulatorPartition = Accumulators<AP, TA, Self::Config>;

    fn execute(
        key_reader: &Self::KeyStage,
        value_reader: &Self::ValueStage,
        query_partition: &Self::QueryPartition,
        key_value_partition: &mut Self::KeyValuePartition,
        softmax_partition: &mut Self::SoftmaxPartition,
        mask: StageMask,
        accumulator_partition: &mut Self::AccumulatorPartition,
        state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        let partition_mask = mask.to_partition(UNIT_POS_Y);

        let p = config.tiling_scheme().partition_size;

        let mut kv = comptime![0u32];

        let mut max_placeholder = TA::init_max_placeholder(config.num_rows_per_unit());
        let mut sum_placeholder = TA::init_sum_placeholder(config.num_rows_per_unit());

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_kv {
            let mut hd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.head_dim {
                let key_tile = SK::tile(key_reader, (hd, kv).runtime());

                TA::fill_key(
                    &key_tile,
                    key_value_partition.get_key_at_mut(hd, kv, config),
                    config.tile_config(),
                );

                comptime![hd += 1];
            }

            let mut q = comptime![0u32];
            let mut scales = Sequence::<RowWise<SM<AP>>>::new();

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

                let state_q = state.index_mut(q);

                scales.push(TA::softmax(
                    softmax_tile,
                    partition_mask.to_tile(q, kv),
                    state_q,
                    &mut max_placeholder,
                    &mut sum_placeholder,
                    config.tiling_scheme().elements_in_partition_head_dim(),
                    config.tile_config(),
                ));

                comptime![q += 1];
            }

            let mut vd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.val_dim {
                let value_tile = SV::tile(value_reader, (kv, vd).runtime());

                TA::fill_value(
                    &value_tile,
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
                let softmax_tile = softmax_partition.get_at(q, kv, config);

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..p.val_dim {
                    TA::accumulate_value(
                        softmax_tile,
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
        state: Sequence<RunningState<SM<AP>>>,
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
                    state.index(q),
                    config.tile_config(),
                );

                comptime![vd += 1];
            }

            comptime![q += 1];
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Sequence<RunningState<SM<AP>>> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(TA::init_state(config.tile_config()));
        }

        sequence
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &Self::AccumulatorPartition,
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
                    Self::AccumulatorPartition::get_at(acc, q, kv, stage_config),
                    stage_config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));

                comptime![kv += 1];
            }

            comptime![q += 1];
        }

        W::on_event(writer, WriteEvent::new_Finish());
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

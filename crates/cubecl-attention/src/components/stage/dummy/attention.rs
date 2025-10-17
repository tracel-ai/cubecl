use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::global::dummy::MaskReader;
use crate::components::global::dummy::QueryReader;
use crate::components::stage::dummy::MaskPartition;
use crate::components::stage::dummy::SoftmaxPartition;
use crate::components::stage::dummy::{Accumulators, DummyStageConfig, KeyValues, QueryPartition};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::RowWise;
use crate::components::tile::RunningState;
use crate::components::tile::TileAttention;
use crate::components::tile::{MaskTile, MaskTileExpand};
use crate::components::tile::{QueryTile, QueryTileExpand};
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

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

    type QueryRegisters = QueryPartition<AP, TA, Self::Config>;
    type KeyValueRegisters = KeyValues<AP, TA, Self::Config>;
    type SoftmaxRegisters = SoftmaxPartition<AP, TA, Self::Config>;
    type AccumulatorRegisters = Accumulators<AP, TA, Self::Config>;
    type MaskRegisters = MaskPartition<AP, TA, Self::Config>;

    fn execute(
        query_partition: &Self::QueryRegisters,
        key_stage: &Self::KeyStage,
        value_stage: &Self::ValueStage,
        key_value_partition: &mut Self::KeyValueRegisters,
        mask_partition: &Self::MaskRegisters,
        softmax_partition: &mut Self::SoftmaxRegisters,
        accumulator_partition: &mut Self::AccumulatorRegisters,
        state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        // let partition_mask = mask.to_partition(UNIT_POS_Y);

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
                let key_tile = SK::tile(key_stage, (hd, kv).runtime());

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

                let mask_tile = mask_partition.get_at(q, kv, config.tiling_scheme());

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
                    mask_tile,
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
                let value_tile = SV::tile(value_stage, (kv, vd).runtime());

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
        acc: &mut Self::AccumulatorRegisters,
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
                    Self::AccumulatorRegisters::get_at_mut(acc, q, vd, config),
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
        acc: &Self::AccumulatorRegisters,
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
                    Self::AccumulatorRegisters::get_at(acc, q, kv, stage_config),
                    stage_config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));

                comptime![kv += 1];
            }

            comptime![q += 1];
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_query(#[comptime] config: Self::Config) -> Self::QueryRegisters {
        Self::QueryRegisters::new(config)
    }

    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValueRegisters {
        Self::KeyValueRegisters::new(config)
    }

    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxRegisters {
        Self::SoftmaxRegisters::new(config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorRegisters {
        Self::AccumulatorRegisters::new(config)
    }

    fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::MaskRegisters {
        Self::MaskRegisters::new(out_of_bounds, config)
    }

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut Self::QueryRegisters,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut q = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut hd = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.head_dim {
                let tile_to_write = registers.get_at_mut(q, hd, config);
                let tile_read = reader.get_tile::<Self::Config>((q, hd).runtime(), config);

                tile_to_write.update(tile_read, config.tile_config());

                comptime![hd += 1];
            }

            comptime![q += 1];
        }
    }

    fn read_mask(
        reader: &MaskReader<AP>,
        registers: &mut Self::MaskRegisters,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut q = comptime![0u32];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..p.seq_q {
            let mut kv = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..p.seq_kv {
                let mask_tile = registers.get_at_mut(q, kv, config.tiling_scheme());

                let (new_origin, tile) = reader.read::<Self::Config>((q, kv), config);
                mask_tile.update(new_origin, tile);

                comptime![kv += 1];
            }

            comptime![q += 1];
        }
    }
}

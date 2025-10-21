use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::fragment::AttentionMatmul;
use crate::components::global::simple::MaskReader;
use crate::components::global::simple::QueryReader;
use crate::components::stage::simple_kv_reuse::MaskPartition;
use crate::components::stage::simple_kv_reuse::SoftmaxPartition;
use crate::components::stage::simple_kv_reuse::{
    AccumulatorPartition, KeyValues, QueryPartition, SimpleKVReuseStageConfig,
};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::RowWise;
use crate::components::tile::RunningState;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

pub struct SimpleKVReuseStageAttention<AP: AttentionPrecision, SK, SV, SO, AM: AttentionMatmul<AP>>
{
    _phantom: PhantomData<(AP, SK, SV, SO, AM)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, ReadOnly, TileKind = Strided>,
    SV: Stage<VS<AP>, ReadOnly, TileKind = Strided>,
    SO: Stage<OS<AP>, ReadWrite, TileKind = Strided>,
    AM: AttentionMatmul<AP>,
> StageAttention<AP> for SimpleKVReuseStageAttention<AP, SK, SV, SO, AM>
{
    type Config = SimpleKVReuseStageConfig<AM::Config>;

    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type QueryRegisters = QueryPartition<AP, AM, Self::Config>;
    type KeyValueRegisters = KeyValues<AP, AM, Self::Config>;
    type SoftmaxRegisters = SoftmaxPartition<AP, AM, Self::Config>;
    type AccumulatorRegisters = AccumulatorPartition<AP, AM, Self::Config>;
    type MaskRegisters = MaskPartition<AP, AM, Self::Config>;

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
        let p = config.tiling_scheme().partition_size;

        let mut max_placeholder =
            TileAttention::<AP, AM>::init_max_placeholder(config.num_rows_per_unit());
        let mut sum_placeholder =
            TileAttention::<AP, AM>::init_sum_placeholder(config.num_rows_per_unit());

        #[unroll]
        for kv in 0..p.seq_kv {
            #[unroll]
            for hd in 0..p.head_dim {
                let key_tile = SK::tile(key_stage, (hd, kv).runtime());

                TileAttention::fill_key(
                    &key_tile,
                    key_value_partition.get_key_at_mut(hd, kv, config),
                    config.tile_config(),
                );
            }

            let mut scales = Sequence::<RowWise<SM<AP>>>::new();

            #[unroll]
            for q in 0..p.seq_q {
                let softmax_tile = softmax_partition.get_at_mut(q, kv, config);
                TileAttention::zero_softmax(softmax_tile, config.tile_config());

                let mask_tile = mask_partition.get_at(q, kv, config.tiling_scheme());

                #[unroll]
                for hd in 0..p.head_dim {
                    let query_tile = query_partition.get_at(q, hd, config);
                    let key_tile = key_value_partition.get_key_at(hd, kv, config);

                    TileAttention::accumulate_score(
                        query_tile,
                        key_tile,
                        softmax_tile,
                        config.tile_config(),
                    );
                }

                let state_q = state.index_mut(q);

                scales.push(TileAttention::softmax(
                    softmax_tile,
                    mask_tile,
                    state_q,
                    &mut max_placeholder,
                    &mut sum_placeholder,
                    config.tiling_scheme().elements_in_partition_head_dim(),
                    config.tile_config(),
                ));
            }

            #[unroll]
            for vd in 0..p.val_dim {
                let value_tile = SV::tile(value_stage, (kv, vd).runtime());

                TileAttention::fill_value(
                    &value_tile,
                    key_value_partition.get_value_at_mut(kv, vd, config),
                    config.tile_config(),
                );
            }

            #[unroll]
            for q in 0..p.seq_q {
                let softmax_tile = softmax_partition.get_at(q, kv, config);

                #[unroll]
                for vd in 0..p.val_dim {
                    TileAttention::accumulate_value(
                        softmax_tile,
                        key_value_partition.get_value_at(kv, vd, config),
                        accumulator_partition.get_at_mut(q, vd, config),
                        scales.index(q),
                        config.tile_config(),
                    );
                }
            }
        }
    }

    fn rescale(
        acc: &mut Self::AccumulatorRegisters,
        state: Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for vd in 0..p.val_dim {
                TileAttention::<AP, AM>::rescale(
                    Self::AccumulatorRegisters::get_at_mut(acc, q, vd, config),
                    state.index(q),
                );
            }
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Sequence<RunningState<SM<AP>>> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(TileAttention::<AP, AM>::init_state(config.tile_config()));
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

        W::on_event(writer, WriteEvent::new_Begin());

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for vd in 0..p.val_dim {
                let tile_pos = (q + UNIT_POS_Y * p.seq_q, vd.runtime());
                let mut tile = Self::OutStage::tile(stage, tile_pos);

                TileAttention::write_results(
                    &mut tile,
                    Self::AccumulatorRegisters::get_at(acc, q, vd, stage_config),
                    stage_config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));
            }
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

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for hd in 0..p.head_dim {
                let tile_to_write = registers.get_at_mut(q, hd, config);
                let tile_read = reader.get_tile::<Self::Config>((q, hd).runtime(), config);

                tile_to_write.update(&tile_read);
            }
        }
    }

    fn read_mask(
        reader: &MaskReader<AP>,
        registers: &mut Self::MaskRegisters,
        #[comptime] config: Self::Config,
    ) {
        let p = config.tiling_scheme().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for kv in 0..p.seq_kv {
                let mask_tile = registers.get_at_mut(q, kv, config.tiling_scheme());

                let (new_origin, tile) = reader.read::<Self::Config>((q, kv), config);
                mask_tile.update(new_origin, tile);
            }
        }
    }
}

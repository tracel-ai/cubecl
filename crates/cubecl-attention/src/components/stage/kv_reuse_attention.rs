use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::stage::StageAttentionConfig;
use crate::components::tile::RowWise;
use crate::components::tile::RunningState;
use crate::components::tile::TileAttention;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{attention_types::*, stage::StageAttention};
use crate::components::{
    fragment::FragmentAttention,
    stage::{KeyValues, QueryPartition, SoftmaxPartition},
};
use crate::components::{global::simple::MaskReader, stage::partitioner::AttentionPartitioner};
use crate::components::{
    global::simple::QueryReader,
    stage::{AccumulatorPartition, MaskPartition},
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
pub struct KVReuseStageAttention<
    AP: AttentionPrecision,
    SK,
    SV,
    SO,
    FA: FragmentAttention<AP>,
    P: AttentionPartitioner,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> {
    #[cube(comptime)]
    _phantom: PhantomData<(AP, SK, SV, SO, FA, P, S)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, ReadOnly, TileKind = Strided>,
    SV: Stage<VS<AP>, ReadOnly, TileKind = Strided>,
    SO: Stage<OS<AP>, ReadWrite, TileKind = Strided>,
    FA: FragmentAttention<AP>,
    P: AttentionPartitioner,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
> StageAttention<AP> for KVReuseStageAttention<AP, SK, SV, SO, FA, P, S>
{
    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = S;
    type Partitioner = P;

    type QueryRegisters = QueryPartition<AP, FA, S>;
    type KeyValueRegisters = KeyValues<AP, FA, S>;
    type SoftmaxRegisters = SoftmaxPartition<AP, FA, S>;
    type AccumulatorRegisters = AccumulatorPartition<AP, FA, S>;
    type MaskRegisters = MaskPartition<AP, FA, S>;

    fn execute(
        query_partition: &QueryPartition<AP, FA, S>,
        key_stage: &SK,
        value_stage: &SV,
        key_value_partition: &mut KeyValues<AP, FA, S>,
        mask_partition: &MaskPartition<AP, FA, S>,
        softmax_partition: &mut SoftmaxPartition<AP, FA, S>,
        accumulator_partition: &mut AccumulatorPartition<AP, FA, S>,
        state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: S,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut max_placeholder =
            TileAttention::<AP, FA>::init_max_placeholder(config.num_rows_per_unit());
        let mut sum_placeholder =
            TileAttention::<AP, FA>::init_sum_placeholder(config.num_rows_per_unit());

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

                scales.push(TileAttention::softmax::<P::Reducer>(
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
        acc: &mut AccumulatorPartition<AP, FA, S>,
        state: Sequence<RunningState<SM<AP>>>,
        #[comptime] config: S,
    ) {
        let p = config.tiling_scheme().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for vd in 0..p.val_dim {
                TileAttention::<AP, FA>::rescale(
                    AccumulatorPartition::<AP, FA, S>::get_at_mut(acc, q, vd, config),
                    state.index(q),
                );
            }
        }
    }

    fn init_state(#[comptime] config: S) -> Sequence<RunningState<SM<AP>>> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(TileAttention::<AP, FA>::init_state(config.tile_config()));
        }

        sequence
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &AccumulatorPartition<AP, FA, S>,
        stage: &mut SO,
        writer: &mut W,
        #[comptime] stage_config: S,
    ) {
        let p = stage_config.tiling_scheme().partition_size;

        W::on_event(writer, WriteEvent::new_Begin());

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for vd in 0..p.val_dim {
                let tile_pos = (q + P::seq_q_index() * p.seq_q, vd.runtime());
                let mut tile = SO::tile(stage, tile_pos);

                TileAttention::<AP, FA>::write_results(
                    &mut tile,
                    AccumulatorPartition::<AP, FA, S>::get_at(acc, q, vd, stage_config),
                    stage_config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_query(#[comptime] config: S) -> QueryPartition<AP, FA, S> {
        QueryPartition::<AP, FA, S>::new(config)
    }

    fn init_key_value(#[comptime] config: S) -> KeyValues<AP, FA, S> {
        KeyValues::<AP, FA, S>::new(config)
    }

    fn init_softmax(#[comptime] config: S) -> SoftmaxPartition<AP, FA, S> {
        SoftmaxPartition::<AP, FA, S>::new(config)
    }

    fn init_accumulator(#[comptime] config: S) -> AccumulatorPartition<AP, FA, S> {
        AccumulatorPartition::<AP, FA, S>::new(config)
    }

    fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: S,
    ) -> MaskPartition<AP, FA, S> {
        MaskPartition::<AP, FA, S>::new(out_of_bounds, config)
    }

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut QueryPartition<AP, FA, S>,
        #[comptime] config: S,
    ) {
        let p = config.tiling_scheme().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for hd in 0..p.head_dim {
                let tile_to_write = registers.get_at_mut(q, hd, config);
                let tile_read = reader.get_tile::<P, S>((q, hd).runtime(), config);

                tile_to_write.update(&tile_read);
            }
        }
    }

    fn read_mask(
        reader: &MaskReader<AP>,
        registers: &mut MaskPartition<AP, FA, S>,
        #[comptime] config: S,
    ) {
        let p = config.tiling_scheme().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for kv in 0..p.seq_kv {
                let mask_tile = registers.get_at_mut(q, kv, config.tiling_scheme());

                let (new_origin, tile) = reader.read::<P, S>((q, kv), config);
                mask_tile.update(new_origin, tile);
            }
        }
    }
}

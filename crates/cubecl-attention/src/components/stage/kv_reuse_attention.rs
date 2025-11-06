use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::tile::RowWise;
use crate::components::{AttentionPrecision, global::GlobalAttentionConfig};
use crate::components::{attention_types::*, stage::StageAttention};
use crate::components::{
    fragment::FragmentAttention,
    stage::{KeyValues, QueryPartition, SoftmaxPartition},
};
use crate::components::{
    fragment::{SoftmaxFragment, SoftmaxFragmentExpand},
    tile::RunningState,
};
use crate::components::{global::simple::MaskReader, stage::partitioner::AttentionPartitioner};
use crate::components::{
    global::simple::QueryReader,
    stage::{AccumulatorPartition, MaskPartition},
};
use crate::components::{stage::StageAttentionConfig, tile::tile_softmax};
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
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut MaskPartition<AP, FA, S>,
        softmax_partition: &mut SoftmaxPartition<AP, FA, S>,
        accumulator_partition: &mut AccumulatorPartition<AP, FA, S>,
        state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: S,
    ) {
        let p = config.tiling_scheme().partition_size;

        let mut max_placeholder = RowWise::new_min_value(config.num_rows_per_unit());
        let mut sum_placeholder = RowWise::new_zero(config.num_rows_per_unit());

        #[unroll]
        for kv in 0..p.seq_kv {
            let mut scales = Sequence::<RowWise<SM<AP>>>::new();

            #[unroll]
            for q in 0..p.seq_q {
                let softmax_tile = softmax_partition.get_at_mut(q);

                FA::zero_softmax(softmax_tile, config.tile_config());

                read_mask::<AP, FA, P, S>((q, kv), mask_reader, mask_partition, config);

                #[unroll]
                for hd in 0..p.head_dim {
                    let query_tile = query_partition.get_at(q, hd, config);
                    let key_tile = key_value_partition.get_key_mut();
                    let key_smem_tile = SK::tile(key_stage, (hd, kv).runtime());

                    FA::fill_key_value(&key_smem_tile, key_tile.key_mut(), config.tile_config());

                    FA::score_matmul(
                        &query_tile.fragment,
                        key_tile.key(),
                        softmax_tile,
                        config.tile_config(),
                    );
                }

                let state_q = state.index_mut(q);

                let softmax_rowwise = softmax_tile.rowwise_mut();

                scales.push(tile_softmax::<AP, FA, P::Reducer>(
                    softmax_rowwise,
                    mask_partition.get_mut(),
                    state_q,
                    &mut max_placeholder,
                    &mut sum_placeholder,
                    config.tiling_scheme().elements_in_partition_head_dim(),
                    config.tile_config(),
                ));

                softmax_tile.update_from_rowwise();

                #[unroll]
                for vd in 0..p.val_dim {
                    let value_smem_tile = SV::tile(value_stage, (kv, vd).runtime());
                    let value_tile = key_value_partition.get_value_mut();

                    FA::fill_key_value(
                        &value_smem_tile,
                        value_tile.value_mut(),
                        config.tile_config(),
                    );

                    let accumulator = accumulator_partition.get_at_mut(q, vd, config);
                    let scale = scales.index(q);

                    accumulator.scale_mul(scale);

                    FA::value_matmul(
                        softmax_tile,
                        key_value_partition.get_value().value(),
                        &mut accumulator.fragment,
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
                AccumulatorPartition::<AP, FA, S>::get_at_mut(acc, q, vd, config)
                    .scale_div(state.index(q).l());
            }
        }
    }

    fn init_state(#[comptime] config: S) -> Sequence<RunningState<SM<AP>>> {
        let p = config.tiling_scheme().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..comptime!(p.seq_q) {
            sequence.push(RunningState::<SM<AP>>::init(config.num_rows_per_unit()));
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

                FA::write_results(
                    &AccumulatorPartition::<AP, FA, S>::get_at(acc, q, vd, stage_config).fragment,
                    &mut tile.slice,
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
}

#[cube]
fn read_mask<
    AP: AttentionPrecision,
    FA: FragmentAttention<AP>,
    P: AttentionPartitioner,
    S: StageAttentionConfig<FragmentAttentionConfig = FA::Config>,
>(
    #[comptime] pos_in_partition: Coords2d,
    reader: &MaskReader<AP>,
    registers: &mut MaskPartition<AP, FA, S>,
    #[comptime] config: S,
) {
    let mask_tile = registers.get_mut();
    let (new_origin, tile) = reader.read::<P, S>(pos_in_partition, config);
    mask_tile.update(new_origin, tile);
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEvent, WriteEventListener},
    stage::Stage,
    tile::io::Strided,
};
use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    global::GlobalAttentionConfig,
    stage::{PartitionAttentionConfig, tile_softmax},
    tile::TileAttentionConfig as _,
};
use crate::components::{attention_types::*, stage::StageAttention};
use crate::components::{global::simple::MaskReader, stage::partitioner::AttentionPartitioner};
use crate::components::{
    global::simple::QueryReader,
    stage::{AccumulatorPartition, MaskPartition},
};
use crate::components::{
    stage::RunningState,
    tile::{FragmentSoftmax, FragmentSoftmaxExpand},
};
use crate::components::{stage::StageAttentionConfig, tile::RowWise};
use crate::components::{
    stage::{KeyValuePartition, QueryPartition, SoftmaxPartition},
    tile::TileAttention,
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
pub struct PartitionAttention<
    AP: AttentionPrecision,
    SK,
    SV,
    SO,
    TA: TileAttention<AP>,
    P: AttentionPartitioner,
> {
    #[cube(comptime)]
    _phantom: PhantomData<(AP, SK, SV, SO, TA, P)>,
}

#[cube]
impl<
    AP: AttentionPrecision,
    SK: Stage<KS<AP>, ReadOnly, TileKind = Strided>,
    SV: Stage<VS<AP>, ReadOnly, TileKind = Strided>,
    SO: Stage<OS<AP>, ReadWrite, TileKind = Strided>,
    TA: TileAttention<AP>,
    P: AttentionPartitioner,
> StageAttention<AP> for PartitionAttention<AP, SK, SV, SO, TA, P>
{
    type KeyStage = SK;
    type ValueStage = SV;
    type OutStage = SO;

    type Config = PartitionAttentionConfig<TA::Config>;
    type Partitioner = P;

    type QueryRegisters = QueryPartition<AP, TA>;
    type KeyValueRegisters = KeyValuePartition<AP, TA>;
    type SoftmaxRegisters = SoftmaxPartition<AP, TA>;
    type AccumulatorRegisters = AccumulatorPartition<AP, TA>;
    type MaskRegisters = MaskPartition<AP, TA>;

    /// Executes the attention computation over one query–key/value partition.
    ///
    /// For each (q, kv) tile pair:
    /// 1. Computes attention scores across the full head dimension for that query row.
    /// 2. Applies masking and softmax locally to obtain unnormalized probabilities.
    /// 3. Uses these probabilities to partially accumulate the corresponding value tiles
    ///    into the output accumulators.
    fn execute(
        query_partition: &QueryPartition<AP, TA>,
        key_stage: &SK,
        value_stage: &SV,
        key_value_partition: &mut KeyValuePartition<AP, TA>,
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut MaskPartition<AP, TA>,
        softmax_partition: &mut SoftmaxPartition<AP, TA>,
        accumulator_partition: &mut AccumulatorPartition<AP, TA>,
        state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;
        let num_rows_per_unit = config.shared().tile_config.num_rows_per_unit();

        // Small working memory in registers
        let mut max_placeholder = RowWise::new_min_value(num_rows_per_unit);
        let mut sum_placeholder = RowWise::new_zero(num_rows_per_unit);

        // The problem is independent on each (q, kv) tile pair
        #[unroll]
        for kv in 0..p.seq_kv {
            #[unroll]
            for q in 0..p.seq_q {
                // Get the q-th softmax tile and zero it
                let softmax_tile = softmax_partition.get_at_mut(q);
                softmax_tile.zero();

                // Get the only mask tile and fill it with q,kv-th data
                let mask_tile = mask_partition.get_mut();
                let (new_origin, mask_data) = mask_reader.read::<P, Self::Config>((q, kv), config);
                mask_tile.update(new_origin, mask_data);

                #[unroll]
                // Iterate over head dim to perform score matmul
                // Contrary to loop for value matmul, all iterations are accumulated into the same tile
                for hd in 0..p.head_dim {
                    // Get the q,hd-th query which is always in registers
                    let query_tile = query_partition.get_at(q, hd, config);

                    // Get the only key-value tile and fill it with hd,kv-th key data
                    let key_tile = key_value_partition.get_key_mut();
                    let key_data = SK::tile(key_stage, (kv, hd).runtime());
                    TA::load_key_transposed(&key_data, key_tile.key_mut(), config.tile_config());

                    // Perform score matmul on query and key, and accumulate in softmax tile
                    TA::score_matmul(
                        &query_tile.fragment,
                        key_tile.key(),
                        softmax_tile,
                        config.tile_config(),
                    );
                }

                // At this point, the softmax tile is filled with score

                // Get the q-th running state, i.e. the one associated with rows from q
                let state_q = state.index_mut(q);

                // Make sure the softmax is in a row-aware layout
                // If the layout is always row-aware, it's a no-op.
                // Otherwise it may go through shared memory
                let softmax_rowwise = softmax_tile.rowwise_mut();

                // Perform the softmax calculation on the (row-format) softmax tile, including masking
                // This mutates the (row-format) softmax tile and the state
                // Also outputs a value needed to scale accumulator later
                let scale = tile_softmax::<AP, TA, P::Reducer>(
                    softmax_rowwise,
                    mask_partition.get(),
                    state_q,
                    &mut max_placeholder,
                    &mut sum_placeholder,
                    p.head_dim * config.tile_config().attention_tile_size().head_dim,
                    config.tile_config(),
                );

                // Make sure the mutations on softmax_rowwise also affect other softmax formats
                softmax_tile.update_from_rowwise();

                // At this point, the softmax tile is filled with probabilities

                #[unroll]
                // Iterate over val dim to perform value matmul
                // Contrary to loop for score matmul, all iterations contribute to different accumulators
                // The same accumulators will be accumulated to at the next kv iteration
                for vd in 0..p.val_dim {
                    // Get the only key-value tile and fill it with hd,kv-th key data
                    let value_data = SV::tile(value_stage, (kv, vd).runtime());
                    let value_tile = key_value_partition.get_value_mut();
                    TA::load_value(&value_data, value_tile.value_mut(), config.tile_config());

                    // Get the q,vd-th accumulator and scale it with previously obtained scale
                    let accumulator = accumulator_partition.get_at_mut(q, vd, config);
                    accumulator.scale_mul(&scale);

                    // Perform value matmul on probabilities and values, and accumulate in accumulators
                    TA::value_matmul(
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
        acc: &mut AccumulatorPartition<AP, TA>,
        state: Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        #[unroll]
        for q in 0..p.seq_q {
            let scale = state.index(q).l();

            #[unroll]
            for vd in 0..p.val_dim {
                AccumulatorPartition::<AP, TA>::get_at_mut(acc, q, vd, config).scale_div(scale);
            }
        }
    }

    fn init_state(#[comptime] config: Self::Config) -> Sequence<RunningState<SM<AP>>> {
        let partition_seq_q = config.shared().partition_size.seq_q;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_seq_q {
            sequence.push(RunningState::<SM<AP>>::init(
                config.shared().tile_config.num_rows_per_unit(),
            ));
        }

        sequence
    }

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &AccumulatorPartition<AP, TA>,
        stage: &mut SO,
        writer: &mut W,
        #[comptime] config: Self::Config,
    ) {
        let p = config.shared().partition_size;

        W::on_event(writer, WriteEvent::new_Begin());

        #[unroll]
        for q in 0..p.seq_q {
            #[unroll]
            for vd in 0..p.val_dim {
                let tile_pos = (q + P::seq_q_index() * p.seq_q, vd.runtime());
                let tile = SO::tile(stage, tile_pos);

                TA::write_results(
                    &AccumulatorPartition::<AP, TA>::get_at(acc, q, vd, config).fragment,
                    &mut tile.as_slice_mut(),
                    config.tile_config(),
                );

                W::on_event(writer, WriteEvent::new_TileStored(tile_pos));
            }
        }

        W::on_event(writer, WriteEvent::new_Finish());
    }

    fn init_query(#[comptime] config: Self::Config) -> QueryPartition<AP, TA> {
        QueryPartition::<AP, TA>::new(config)
    }

    fn init_key_value(#[comptime] config: Self::Config) -> KeyValuePartition<AP, TA> {
        KeyValuePartition::<AP, TA>::new(config)
    }

    fn init_softmax(#[comptime] config: Self::Config) -> SoftmaxPartition<AP, TA> {
        SoftmaxPartition::<AP, TA>::new(config)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> AccumulatorPartition<AP, TA> {
        AccumulatorPartition::<AP, TA>::new(config)
    }

    fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> MaskPartition<AP, TA> {
        MaskPartition::<AP, TA>::new(out_of_bounds, config)
    }

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut QueryPartition<AP, TA>,
        #[comptime] config: Self::Config,
    ) {
        let partition_seq_q = config.shared().partition_size.seq_q;
        let partition_head_dim = config.shared().partition_size.head_dim;
        let attention_tile_size = config.shared().tile_config.attention_tile_size();

        #[unroll]
        for q in 0..partition_seq_q {
            #[unroll]
            for hd in 0..partition_head_dim {
                let tile_to_write = registers.get_at_mut(q, hd, config);
                let tile_read = reader.get_tile::<P>(
                    (q, hd).runtime(),
                    attention_tile_size,
                    partition_seq_q,
                    partition_head_dim,
                );

                tile_to_write.update(&tile_read);
            }
        }
    }
}

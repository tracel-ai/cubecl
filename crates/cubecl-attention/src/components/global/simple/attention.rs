use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::PartitionedStage;
use cubecl_matmul::components::stage::StridedStage;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::simple::reader::{AttentionReader, AttentionReaderExpand};
use crate::components::global::simple::{AttentionWriter, AttentionWriterExpand, MaskReader};
use crate::components::global::{
    AttentionGlobalLayout,
    simple::{DummyKeyReader, DummyValueReader},
};
use crate::components::stage::{AttentionPartitioner, AttentionTilingLayout, StageAttention};
use crate::components::{AttentionIdent, global::simple::QueryReader};
use crate::components::{
    AttentionPrecision,
    global::{GlobalAttention, simple::config::SimpleGlobalConfig},
};

pub struct SimpleGlobalAttention<AP: AttentionPrecision, SA: StageAttention<AP>> {
    _phantom: PhantomData<(AP, SA)>,
}

#[cube]
impl<
    SA: StageAttention<
            AP,
            KeyStage = StridedStage<KS<AP>, AttentionTilingLayout>,
            ValueStage = StridedStage<VS<AP>, AttentionTilingLayout>,
            OutStage = PartitionedStage<OS<AP>>,
        >,
    AP: AttentionPrecision,
> GlobalAttention<AP> for SimpleGlobalAttention<AP, SA>
{
    type KeyReader = DummyKeyReader<AP, Self::Config>;
    type ValueReader = DummyValueReader<AP, Self::Config>;
    type MaskReader = MaskReader<AP>;

    type Writer = <SA::Partitioner as AttentionPartitioner>::Writer<OS<AP>, OG<AP>>;

    type Config = SimpleGlobalConfig<SA::Config>;

    fn execute(
        query_reader: QueryReader<AP>,
        mut key_reader: Self::KeyReader,
        mut value_reader: Self::ValueReader,
        mut mask_reader: Self::MaskReader,
        mut writer: Self::Writer,
        seq_q: u32,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    ) {
        // Init staging shared memories
        let mut key_stage = key_reader.init_stage(config);
        let mut value_stage = value_reader.init_stage(config);

        // Load queries which stay alive in registers for all the kernel
        let mut query_registers = SA::init_query(config.stage_config());
        SA::read_query(&query_reader, &mut query_registers, config.stage_config());

        // Init registers that will change inside global loop
        let mut key_value_registers = SA::init_key_value(config.stage_config());
        let mut mask_registers =
            SA::init_mask(CubeOption::new_Some((seq_q, seq_kv)), config.stage_config());
        let mut softmax_registers = SA::init_softmax(config.stage_config());
        let mut accumulator_registers = SA::init_accumulator(config.stage_config());

        // Init running state
        let mut stage_state = SA::init_state(config.stage_config());

        // Define number of global iterations
        let num_stage_iterations =
            seq_kv.div_ceil(config.tiling_scheme().elements_in_partition_seq_kv());

        // Global loop over seq_kv
        for _ in 0..num_stage_iterations {
            // Put key and value into stage
            key_reader.read_global(&mut key_stage, config);
            value_reader.read_global(&mut value_stage, config);

            sync_cube();

            // Core of flash attention
            SA::execute(
                &query_registers,
                &key_stage,
                &value_stage,
                &mut key_value_registers,
                &mask_reader,
                &mut mask_registers,
                &mut softmax_registers,
                &mut accumulator_registers,
                &mut stage_state,
                config.stage_config(),
            );

            sync_cube();

            // Advance in seq_kv direction
            key_reader.advance_view();
            value_reader.advance_view();
            mask_reader.advance_view();
        }

        // Accumulators must be rescaled using running state
        SA::rescale(
            &mut accumulator_registers,
            stage_state,
            config.stage_config(),
        );

        // Write accumulators to output
        let mut out_stage = writer.stage();
        SA::write::<Self::Writer, Self::Config>(
            &accumulator_registers,
            &mut out_stage,
            &mut writer,
            config.stage_config(),
        )
    }

    fn init_query_reader(
        batch_index: u32,
        stage_q_offset: u32,
        query: VirtualTensor<QG<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<AP> {
        let layout = AttentionGlobalLayout::new(
            &query,
            batch_index,
            config.global_memory_config(AttentionIdent::Query),
        );

        QueryReader::<AP>::new(stage_q_offset, query.view(layout))
    }

    fn init_key_reader(
        batch_index: u32,
        key: VirtualTensor<KG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader {
        let step = reduction_step::<Self::Config>(config);
        let layout = AttentionGlobalLayout::new(
            &key,
            batch_index,
            config.global_memory_config(AttentionIdent::Key),
        );
        DummyKeyReader::new(key.view(layout), step)
    }

    fn init_value_reader(
        batch_index: u32,
        value: VirtualTensor<VG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader {
        let step = reduction_step::<Self::Config>(config);
        let layout = AttentionGlobalLayout::new(
            &value,
            batch_index,
            config.global_memory_config(AttentionIdent::Value),
        );
        DummyValueReader::new(value.view(layout), step)
    }

    fn init_mask_reader(
        batch_index: u32,
        stage_q_offset: u32,
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        seq_kv_shape: u32,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader {
        let step = reduction_step::<Self::Config>(config);
        let partition_q_offset = <SA::Partitioner as AttentionPartitioner>::seq_q_index()
            * config.tiling_scheme().elements_in_partition_seq_q();

        match mask {
            CubeOption::Some(mask) => {
                let layout = AttentionGlobalLayout::new(
                    &mask,
                    batch_index,
                    config.global_memory_config(AttentionIdent::Value),
                );

                MaskReader::new_materialized(
                    stage_q_offset,
                    partition_q_offset,
                    mask.view(layout),
                    step,
                    seq_kv_shape,
                )
            }
            CubeOption::None => MaskReader::new_logical(stage_q_offset + partition_q_offset, step),
        }
    }

    fn init_writer(
        batch_index: u32,
        stage_q_offset: u32,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let conf = config.global_memory_config(AttentionIdent::Out);
        let layout = AttentionGlobalLayout::new(&out, batch_index, conf);
        let out = out.view_mut(layout);

        Self::Writer::new::<SA::Config>(
            out.slice_mut_unchecked((stage_q_offset, 0), out.shape()),
            conf,
            config.stage_config(),
        )
    }
}

#[cube]
fn reduction_step<C: GlobalAttentionConfig>(#[comptime] config: C) -> u32 {
    config
        .tiling_scheme()
        .elements_in_partition_seq_kv()
        .runtime()
}

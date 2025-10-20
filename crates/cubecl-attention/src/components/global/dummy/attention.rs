use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::PartitionedStage;
use cubecl_matmul::components::stage::StridedStage;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::attention_types::*;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::dummy::MaskReader;
use crate::components::global::dummy::reader::{AttentionReader, AttentionReaderExpand};
use crate::components::global::dummy::writer::DummyWriter;
use crate::components::global::{
    AttentionGlobalLayout,
    dummy::{DummyKeyReader, DummyValueReader},
};
use crate::components::stage::StageAttention;
use crate::components::tile::AttentionTilingLayout;
use crate::components::{AttentionIdent, global::dummy::QueryReader};
use crate::components::{
    AttentionPrecision,
    global::{GlobalAttention, dummy::config::DummyGlobalConfig},
};

pub struct DummyGlobalAttention<AP: AttentionPrecision, SA: StageAttention<AP>> {
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
> GlobalAttention<AP> for DummyGlobalAttention<AP, SA>
{
    type KeyReader = DummyKeyReader<AP, Self::Config>;
    type ValueReader = DummyValueReader<AP, Self::Config>;
    type MaskReader = MaskReader<AP>;

    type Writer = DummyWriter<(OG<AP>, OS<AP>)>;

    type Config = DummyGlobalConfig<SA::Config>;

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
        let mut key_stage = key_reader.init_stage(config);
        let mut value_stage = value_reader.init_stage(config);

        let mut query_registers = SA::init_query(config.stage_config());
        let mut key_value_registers = SA::init_key_value(config.stage_config());
        let mut mask_registers =
            SA::init_mask(CubeOption::new_Some((seq_q, seq_kv)), config.stage_config());
        let mut softmax_registers = SA::init_softmax(config.stage_config());
        let mut accumulator_registers = SA::init_accumulator(config.stage_config());

        let mut stage_state = SA::init_state(config.stage_config());

        let seq_kv_stage = config.tiling_scheme().elements_in_partition_seq_kv();

        let num_stage_iterations = seq_kv.div_ceil(seq_kv_stage);

        SA::read_query(&query_reader, &mut query_registers, config.stage_config());

        for _ in 0..num_stage_iterations {
            key_reader.read_global(&mut key_stage, config);
            value_reader.read_global(&mut value_stage, config);

            SA::read_mask(&mask_reader, &mut mask_registers, config.stage_config());

            sync_cube();

            SA::execute(
                &query_registers,
                &key_stage,
                &value_stage,
                &mut key_value_registers,
                &mask_registers,
                &mut softmax_registers,
                &mut accumulator_registers,
                &mut stage_state,
                config.stage_config(),
            );

            sync_cube();

            key_reader.advance_view();
            value_reader.advance_view();
            mask_reader.advance_view();
        }

        SA::rescale(
            &mut accumulator_registers,
            stage_state,
            config.stage_config(),
        );

        let mut out_stage = writer.stage();

        SA::write::<Self::Writer, Self::Config>(
            &accumulator_registers,
            &mut out_stage,
            &mut writer,
            config.stage_config(),
        )
    }

    fn init_query_reader(
        q_offset: u32,
        query: VirtualTensor<QG<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<AP> {
        let layout = AttentionGlobalLayout::new(
            &query,
            0,
            config.global_memory_config(AttentionIdent::Query),
        );

        QueryReader::<AP>::new(q_offset, query.view(layout))
    }

    fn init_key_reader(
        key: VirtualTensor<KG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader {
        let step = reduction_step::<Self::Config>(config);
        let layout =
            AttentionGlobalLayout::new(&key, 0, config.global_memory_config(AttentionIdent::Key));
        DummyKeyReader::new(key.view(layout), step)
    }

    fn init_value_reader(
        value: VirtualTensor<VG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader {
        let step = reduction_step::<Self::Config>(config);
        let layout = AttentionGlobalLayout::new(
            &value,
            0,
            config.global_memory_config(AttentionIdent::Value),
        );
        DummyValueReader::new(value.view(layout), step)
    }

    fn init_mask_reader(
        q_offset: u32,
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        seq_kv_shape: u32,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader {
        let step = reduction_step::<Self::Config>(config);

        match mask {
            CubeOption::Some(mask) => {
                let layout = AttentionGlobalLayout::new(
                    &mask,
                    0,
                    config.global_memory_config(AttentionIdent::Value),
                );

                MaskReader::new_materialized(q_offset, mask.view(layout), step, seq_kv_shape)
            }
            CubeOption::None => MaskReader::new_logical(q_offset, step),
        }
    }

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let conf = config.global_memory_config(AttentionIdent::Out);
        let layout = AttentionGlobalLayout::new(&out, 0, conf);
        let out = out.view_mut(layout);

        Self::Writer::new::<SA::Config>(
            out.slice_mut_unchecked((q_offset, 0), out.shape()),
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

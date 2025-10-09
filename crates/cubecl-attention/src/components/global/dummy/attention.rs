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
use crate::components::{GlobalMask, LogicalMask};

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
    type MaskReader = CubeOption<MaskReader<AP, Self::Config>>;

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
        let key_stage = key_reader.stage();
        let value_stage = value_reader.stage();

        let mut stage_state = SA::init_state(config.stage_config());

        let (query, mut key_value, mut softmax, mut accumulator) =
            SA::init_partitions(query_reader, config.stage_config());

        let seq_kv_stage = config.tiling_scheme().elements_in_partition_seq_kv();

        let num_stage_iterations = seq_kv.div_ceil(seq_kv_stage);

        let logical_mask = LogicalMask {
            causal: false,
            out_of_bounds: CubeOption::new_Some((seq_q, seq_kv)),
        };
        let mask = GlobalMask::new(logical_mask, config.tiling_scheme());

        for i in 0..num_stage_iterations {
            key_reader.read_transposed(config);
            value_reader.read(config);
            sync_cube();

            SA::execute(
                &key_stage,
                &value_stage,
                &query,
                &mut key_value,
                &mut softmax,
                mask.to_stage(CUBE_POS, i),
                &mut accumulator,
                &mut stage_state,
                config.stage_config(),
            );

            sync_cube();
            key_reader.advance_view();
            value_reader.advance_view();
        }

        SA::rescale(&mut accumulator, stage_state, config.stage_config());

        let mut out_stage = writer.stage();

        SA::write::<Self::Writer, Self::Config>(
            &accumulator,
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
        DummyKeyReader::new(key.view(layout), step, config)
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
        DummyValueReader::new(value.view(layout), step, config)
    }

    fn init_mask_reader(
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader {
        let step = reduction_step::<Self::Config>(config);

        // TODO this is a simplification for now
        match mask {
            CubeOption::Some(mask) => {
                let layout = AttentionGlobalLayout::new(
                    &mask,
                    0,
                    config.global_memory_config(AttentionIdent::Value),
                );

                CubeOption::new_Some(MaskReader::new(mask.view(layout), step, config))
            }
            CubeOption::None => CubeOption::new_None(),
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

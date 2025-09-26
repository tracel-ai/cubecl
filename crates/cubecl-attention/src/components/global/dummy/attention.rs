use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::StridedStage;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::components::GlobalMask;
use crate::components::attention_types::*;
use crate::components::global::base::GlobalAttentionConfig;
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
        >,
    AP: AttentionPrecision,
> GlobalAttention<AP> for DummyGlobalAttention<AP, SA>
{
    type KeyReader = DummyKeyReader<AP, Self::Config>;
    type ValueReader = DummyValueReader<AP, Self::Config>;

    type Writer = SA::Writer;

    type Config = DummyGlobalConfig<SA::Config>;

    fn execute(
        query_reader: QueryReader<AP>,
        mut key_reader: Self::KeyReader,
        mut value_reader: Self::ValueReader,
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
        let mask = GlobalMask::new(seq_q, seq_kv, config.tiling_scheme());

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

        SA::write::<Self::Config>(&accumulator, &mut writer, config.stage_config(), config)
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

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let conf = config.global_memory_config(AttentionIdent::Out);
        let layout = AttentionGlobalLayout::new(&out, 0, conf);
        let out = out.view_mut(layout);
        SA::init_writer(out.slice_mut_unchecked((q_offset, 0), out.shape()), conf)
    }
}

#[cube]
fn reduction_step<C: GlobalAttentionConfig>(#[comptime] config: C) -> u32 {
    config
        .tiling_scheme()
        .elements_in_partition_seq_kv()
        .runtime()
}

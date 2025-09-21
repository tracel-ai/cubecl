use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::SimpleGlobalLayout;
use cubecl_matmul::components::stage::FullStageReader;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

use crate::components::FlashIdent;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::dummy::load::{DummyKeyReader, DummyQueryReader, DummyValueReader};
use crate::components::stage::{StageAttention, StageAttentionConfig};
use crate::components::tile::AttentionTilingLayout;
use crate::components::tile::dummy::FlashMatmulConfig;
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
            KeyReader = FullStageReader<AP::ES, AttentionTilingLayout>,
            ValueReader = FullStageReader<AP::ES, AttentionTilingLayout>,
        >,
    AP: AttentionPrecision,
> GlobalAttention<AP> for DummyGlobalAttention<AP, SA>
{
    type KeyReader = DummyKeyReader<AP, Self::Config>;
    type ValueReader = DummyValueReader<AP, Self::Config>;

    type Writer = SA::Writer;

    type Config = DummyGlobalConfig<SA::Config>;

    fn execute(
        query_reader: DummyQueryReader<AP, Self::Config>,
        mut key_reader: Self::KeyReader,
        mut value_reader: Self::ValueReader,
        mut writer: Self::Writer,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    ) {
        comment!("Global: Execute");

        let query_reader = query_reader.stage_reader(config);
        let key_stage_reader = key_reader.stage_reader();
        let value_stage_reader = value_reader.stage_reader();

        let mut stage_state = SA::init_state(config.stage_config());

        let (query, mut key_value, mut score_prob, mut accumulator) =
            SA::init_fragments(query_reader, config.stage_config());

        let seq_kv_tile = config
            .stage_config()
            .tile_config()
            .attention_tile_size()
            .seq_kv;

        let seq_q_tile = config
            .stage_config()
            .tile_config()
            .attention_tile_size()
            .seq_q;

        let num_stage_iterations = div_ceil(seq_kv, seq_kv_tile);

        for i in 0..num_stage_iterations {
            let out_of_bounds_mask = if config.stage_config().tile_config().check_bounds() {
                CubeOption::new_Some((seq_q_tile, seq_kv - i * seq_kv_tile))
            } else {
                CubeOption::new_None()
            };

            key_reader.load_transposed(config);
            value_reader.load(config);
            sync_cube();

            SA::execute(
                &key_stage_reader,
                &value_stage_reader,
                &query,
                &mut key_value,
                &mut score_prob,
                &mut accumulator,
                &mut stage_state,
                out_of_bounds_mask,
                config.stage_config(),
            );

            sync_cube();
            comment!("Advance view");
            key_reader.advance_view();
            value_reader.advance_view();
        }

        SA::rescale(&mut accumulator, stage_state, config.stage_config());

        SA::write::<Self::Config>(&accumulator, &mut writer, config.stage_config(), config)
    }

    fn init_query_reader(
        q_offset: u32,
        query: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> DummyQueryReader<AP, Self::Config> {
        comment!("Global: Init Query Reader");
        let layout =
            SimpleGlobalLayout::new(&query, 0, config.global_memory_config(FlashIdent::Query));
        DummyQueryReader::<AP, Self::Config>::new(q_offset, query.view(layout), config)
    }

    fn init_key_reader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader {
        comment!("Global: Init Key Reader");
        let layout = SimpleGlobalLayout::new(&key, 0, config.global_memory_config(FlashIdent::Key));
        let k_step = k_step::<Self::Config>(config);
        DummyKeyReader::new(key.view(layout), k_step, config)
    }

    fn init_value_reader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader {
        comment!("Global: Init Value Reader");
        let layout =
            SimpleGlobalLayout::new(&value, 0, config.global_memory_config(FlashIdent::Value));
        let k_step = k_step::<Self::Config>(config);
        DummyValueReader::new(value.view(layout), k_step, config)
    }

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<AP::EO, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        comment!("Global: Init Writer");
        let conf = config.global_memory_config(FlashIdent::Out);
        let layout = SimpleGlobalLayout::new(&out, 0, conf);
        let out = out.view_mut(layout);
        SA::init_writer(out.slice_mut_unchecked((q_offset, 0), out.shape()), conf)
    }
}

#[cube]
fn k_step<C: GlobalAttentionConfig>(#[comptime] config: C) -> u32 {
    config
        .stage_config()
        .tile_config()
        .attention_tile_size()
        .seq_kv
        .runtime()
}

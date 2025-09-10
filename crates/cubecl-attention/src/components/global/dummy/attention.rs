use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::SimpleGlobalLayout;
use cubecl_matmul::components::stage::FullStageToTileReader;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

use crate::components::FlashIdent;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::dummy::load::{DummyKeyLoader, DummyQueryLoader, DummyValueLoader};
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
            KeyReader = FullStageToTileReader<AP::ES, AttentionTilingLayout>,
            ValueReader = FullStageToTileReader<AP::ES, AttentionTilingLayout>,
        >,
    AP: AttentionPrecision,
> GlobalAttention<AP> for DummyGlobalAttention<AP, SA>
{
    type KeyLoader = DummyKeyLoader<AP, Self::Config>;
    type ValueLoader = DummyValueLoader<AP, Self::Config>;

    type Writer = SA::Writer;

    type Config = DummyGlobalConfig<SA::Config>;

    fn execute(
        query_loader: DummyQueryLoader<AP, Self::Config>,
        mut key_loader: Self::KeyLoader,
        mut value_loader: Self::ValueLoader,
        mut writer: Self::Writer,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    ) {
        comment!("Global: Execute");

        let query_reader = query_loader.reader(config);
        let key_reader = key_loader.reader();
        let value_reader = value_loader.reader();

        let mut stage_state = SA::init_state(config.stage_config());

        let (query, mut key_value, mut score_prob, mut accumulator) =
            SA::init_fragments(query_reader, config.stage_config());

        let seq_kv_stage = config.tiling_scheme().seq_kv();

        let seq_q_tile = config
            .stage_config()
            .tile_config()
            .attention_tile_size()
            .seq_q;

        let num_stage_iterations = div_ceil(seq_kv, seq_kv_stage);

        for i in 0..num_stage_iterations {
            let out_of_bounds_mask = if config.stage_config().tile_config().check_bounds() {
                CubeOption::new_Some((seq_q_tile, seq_kv - i * seq_kv_stage))
            } else {
                CubeOption::new_None()
            };

            key_loader.load_transposed(config);
            value_loader.load(config);
            sync_cube();

            SA::execute(
                &key_reader,
                &value_reader,
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
            key_loader.advance_view(seq_kv_stage);
            value_loader.advance_view(seq_kv_stage);
        }

        SA::rescale(&mut accumulator, stage_state, config.stage_config());

        SA::write::<Self::Config>(&accumulator, &mut writer, config.stage_config(), config)
    }

    fn init_query_loader(
        q_offset: u32,
        query: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> DummyQueryLoader<AP, Self::Config> {
        comment!("Global: Init Query Loader");
        let layout =
            SimpleGlobalLayout::new(&query, config.global_memory_config(FlashIdent::Query));
        DummyQueryLoader::<AP, Self::Config>::new(q_offset, query.view(layout.virt()), config)
    }

    fn init_key_loader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyLoader {
        comment!("Global: Init Key Loader");
        let layout = SimpleGlobalLayout::new(&key, config.global_memory_config(FlashIdent::Key));
        DummyKeyLoader::new(key.view(layout.virt()), config)
    }

    fn init_value_loader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueLoader {
        comment!("Global: Init Value Loader");
        let layout =
            SimpleGlobalLayout::new(&value, config.global_memory_config(FlashIdent::Value));
        DummyValueLoader::new(value.view(layout.virt()), config)
    }

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<AP::EO, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        comment!("Global: Init Writer");
        let layout = SimpleGlobalLayout::new(&out, config.global_memory_config(FlashIdent::Out));
        SA::init_writer(q_offset, out.view_mut(layout.virt()))
    }
}

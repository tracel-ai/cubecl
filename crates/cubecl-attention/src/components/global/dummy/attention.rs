use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::FullStageReader;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

use crate::components::FlashIdent;
use crate::components::global::AttentionGlobalLayout;
use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::dummy::load::{DummyKeyLoader, DummyValueLoader, QueryLoader};
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
    type KeyLoader = DummyKeyLoader<AP, Self::Config>;
    type ValueLoader = DummyValueLoader<AP, Self::Config>;

    type Writer = SA::Writer;

    type Config = DummyGlobalConfig<SA::Config>;

    fn execute(
        query_loader: QueryLoader<AP>,
        mut key_loader: Self::KeyLoader,
        mut value_loader: Self::ValueLoader,
        mut writer: Self::Writer,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    ) {
        let key_reader = key_loader.reader();
        let value_reader = value_loader.reader();

        let mut stage_state = SA::init_state(config.stage_config());

        let (query, mut key_value, mut score_prob, mut accumulator) =
            SA::init_fragments(query_loader, config.stage_config());

        let seq_kv_stage = config.tiling_scheme().seq_kv();

        let num_stage_iterations = div_ceil(seq_kv, seq_kv_stage.runtime());

        for i in 0..num_stage_iterations {
            let out_of_bounds_mask = if config.stage_config().tile_config().check_bounds() {
                let seq_q_stage = config.stage_config().tiling_scheme().seq_q();
                CubeOption::new_Some((seq_q_stage, seq_kv - i * seq_kv_stage))
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
    ) -> QueryLoader<AP> {
        let layout =
            AttentionGlobalLayout::new(&query, 0, config.global_memory_config(FlashIdent::Query));

        QueryLoader::<AP>::new(q_offset, query.view(layout))
    }

    fn init_key_loader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyLoader {
        let layout =
            AttentionGlobalLayout::new(&key, 0, config.global_memory_config(FlashIdent::Key));
        DummyKeyLoader::new(key.view(layout), config)
    }

    fn init_value_loader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueLoader {
        let layout =
            AttentionGlobalLayout::new(&value, 0, config.global_memory_config(FlashIdent::Value));
        DummyValueLoader::new(value.view(layout), config)
    }

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<AP::EO, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer {
        let layout =
            AttentionGlobalLayout::new(&out, 0, config.global_memory_config(FlashIdent::Out));
        let out = out.view_mut(layout);
        SA::init_writer(out.slice_mut_unchecked((q_offset, 0), out.shape()))
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::FullStageToTileReader;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
use crate::components::global::dummy::load::{DummyKeyLoader, DummyQueryLoader, DummyValueLoader};
use crate::components::stage::AttentionTilingLayout;
use crate::components::{
    AttentionPrecision,
    global::{GlobalAttention, dummy::config::DummyGlobalConfig},
    stage::StageAttention,
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
        query_loader: DummyQueryLoader<AP>,
        mut key_loader: Self::KeyLoader,
        mut value_loader: Self::ValueLoader,
        mut writer: Self::Writer,
        #[comptime] config: Self::Config,
    ) {
        comment!("Global: Execute");

        let query_reader = query_loader.reader();
        let key_reader = key_loader.reader();
        let value_reader = value_loader.reader();

        let mut stage_state = SA::init_state(config.stage_config());

        let (query, mut key_value, mut score_prob, mut accumulator) =
            SA::init_fragments(query_reader, config.stage_config());

        for _ in 0..config.tc() {
            key_loader.load_transposed();
            value_loader.load();
            SA::execute(
                &key_reader,
                &value_reader,
                &query,
                &mut key_value,
                &mut score_prob,
                &mut accumulator,
                &mut stage_state,
                config.stage_config(),
            );
        }

        SA::rescale(&mut accumulator, stage_state, config.stage_config());

        SA::write::<Self::Config>(&accumulator, &mut writer, config.stage_config(), config)
    }

    fn init_query_loader(query: VirtualTensor<AP::EI>) -> DummyQueryLoader<AP> {
        comment!("Global: Init Query Loader");
        DummyQueryLoader::<AP>::new(query)
    }

    fn init_key_loader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyLoader {
        comment!("Global: Init Key Loader");
        DummyKeyLoader::new(key, config)
    }

    fn init_value_loader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueLoader {
        comment!("Global: Init Value Loader");
        DummyValueLoader::new(value, config)
    }

    fn init_writer(out: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer {
        comment!("Global: Init Writer");
        SA::init_writer(out)
    }
}

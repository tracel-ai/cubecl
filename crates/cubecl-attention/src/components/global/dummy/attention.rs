use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::{GlobalMemoryConfig, TensorReader};
use cubecl_matmul::components::stage::{FullStageToTileReader, StageMemory};
use cubecl_matmul::components::{MatrixLayout, StageIdent};
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
    type QueryLoader = DummyQueryLoader<AP>;
    type KeyLoader = DummyKeyLoader<AP, Self::Config>;
    type ValueLoader = DummyValueLoader<AP, Self::Config>;

    type Writer = SA::Writer;
    type Accumulator = SA::Accumulator;

    type Config = DummyGlobalConfig<SA::Config>;

    fn execute(
        query_loader: Self::QueryLoader,
        mut key_loader: Self::KeyLoader,
        mut value_loader: Self::ValueLoader,
        writer: Self::Writer,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        comment!("Global: Execute");
        SA::zero_accumulator(acc);

        let query_reader = query_loader.reader();
        query_loader.load();

        let key_reader = key_loader.reader();
        let value_reader = value_loader.reader();

        let mut stage_state = SA::init_state(config.stage_config());

        for j in 0..config.tc() {
            key_loader.load();
            value_loader.load();
            SA::execute(
                &query_reader,
                &key_reader,
                &value_reader,
                acc,
                &mut stage_state,
                config.stage_config(),
            );
        }

        SA::last_update(acc, stage_state);

        SA::write(acc, writer)
    }

    fn init_query_loader(query: VirtualTensor<AP::EI>) -> Self::QueryLoader {
        comment!("Global: Init Query Loader");
        DummyQueryLoader::new(query)
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

    fn init_accumulator() -> Self::Accumulator {
        comment!("Global: Init Accumulator");
        SA::init_accumulator()
    }
}

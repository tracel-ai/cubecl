use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::StageIdent;
use cubecl_matmul::components::global::global_memory::TensorReader;
use cubecl_matmul::components::stage::{FullStageToTileReader, StageMemory};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::components::global::base::GlobalAttentionConfig;
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
        key_loader: Self::KeyLoader,
        value_loader: Self::ValueLoader,
        writer: Self::Writer,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        SA::zero_accumulator(acc);

        let query_reader = query_loader.reader();
        query_loader.load();

        let key_reader = key_loader.reader();
        let value_reader = value_loader.reader();

        let mut stage_state = SA::init_state();

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
        DummyQueryLoader::new(query)
    }

    fn init_key_loader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyLoader {
        DummyKeyLoader::new(key, config)
    }

    fn init_value_loader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueLoader {
        DummyValueLoader::new(value, config)
    }

    fn init_writer() -> Self::Writer {
        SA::init_writer()
    }

    fn init_accumulator() -> Self::Accumulator {
        SA::init_accumulator()
    }
}

#[derive(CubeType)]
pub struct DummyQueryLoader<AP: AttentionPrecision> {
    tensor_reader: TensorReader<AP::EI>,

    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}
#[derive(CubeType)]
pub struct DummyKeyLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<(AP, G)>,
}
#[derive(CubeType)]
pub struct DummyValueLoader<AP: AttentionPrecision, G: GlobalAttentionConfig> {
    tensor_reader: TensorReader<AP::EI>,
    stage_memory: StageMemory<AP::ES, AttentionTilingLayout>,

    #[cube(comptime)]
    _phantom: PhantomData<(AP, G)>,
}

#[cube]
impl<AP: AttentionPrecision> DummyQueryLoader<AP> {
    fn new(query: VirtualTensor<AP::EI>) -> Self {
        let tensor_reader = TensorReader::new(query, 0, 0, 0);

        DummyQueryLoader::<AP> {
            tensor_reader,
            _phantom: PhantomData,
        }
    }

    fn reader(&self) -> RegisterToTileReader {
        RegisterToTileReader {}
    }

    fn load(&self) {}
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyKeyLoader<AP, G> {
    fn new(key: VirtualTensor<AP::EI>, #[comptime] config: G) -> Self {
        let tensor_reader = TensorReader::new(key, 0, 0, 0);
        let stage_memory = StageMemory::new::<G::ScoreStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.score_stage_memory_config(),
        );

        DummyKeyLoader::<AP, G> {
            tensor_reader,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    fn load(&self) {}
}

#[cube]
impl<AP: AttentionPrecision, G: GlobalAttentionConfig> DummyValueLoader<AP, G> {
    fn new(value: VirtualTensor<AP::EI>, #[comptime] config: G) -> Self {
        let tensor_reader = TensorReader::new(value, 0, 0, 0);
        let stage_memory = StageMemory::new::<G::ValueStageMemoryConfig>(
            1u32,
            StageIdent::Rhs,
            config.value_stage_memory_config(),
        );

        DummyValueLoader::<AP, G> {
            tensor_reader,
            stage_memory,
            _phantom: PhantomData,
        }
    }

    fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: self.stage_memory,
            stage_ident: StageIdent::Rhs,
        }
    }

    fn load(&self) {}
}

#[derive(CubeType)]
pub struct RegisterToTileReader {}

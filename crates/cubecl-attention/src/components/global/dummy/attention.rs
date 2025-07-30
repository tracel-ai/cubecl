use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::StageIdent;
use cubecl_matmul::components::stage::FullStageToTileReader;
use std::marker::PhantomData;

use crate::components::global::base::GlobalConfig;
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
    type KeyLoader = DummyKeyLoader<AP>;
    type ValueLoader = DummyValueLoader<AP>;

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

    fn init_query_loader() -> Self::QueryLoader {
        todo!()
    }

    fn init_key_loader() -> Self::KeyLoader {
        todo!()
    }

    fn init_value_loader() -> Self::ValueLoader {
        todo!()
    }

    fn init_writer() -> Self::Writer {
        todo!()
    }

    fn init_accumulator() -> Self::Accumulator {
        todo!()
    }
}

#[derive(CubeType)]
pub struct DummyWriter {}
#[derive(CubeType)]
pub struct DummyAccumulator {}

#[derive(CubeType)]
pub struct DummyQueryLoader<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}
#[derive(CubeType)]
pub struct DummyKeyLoader<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}
#[derive(CubeType)]
pub struct DummyValueLoader<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}

#[cube]
impl<AP: AttentionPrecision> DummyQueryLoader<AP> {
    fn reader(&self) -> GlobalToTileReader {
        GlobalToTileReader {}
    }

    fn load(&self) {}
}

#[cube]
impl<AP: AttentionPrecision> DummyKeyLoader<AP> {
    fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: todo!(),
            stage_ident: StageIdent::Rhs,
        }
    }

    fn load(&self) {}
}

#[cube]
impl<AP: AttentionPrecision> DummyValueLoader<AP> {
    fn reader(&self) -> FullStageToTileReader<AP::ES, AttentionTilingLayout> {
        FullStageToTileReader::<AP::ES, AttentionTilingLayout> {
            stage_memory: todo!(),
            stage_ident: StageIdent::Rhs,
        }
    }

    fn load(&self) {}
}

#[cube]
impl DummyAccumulator {
    fn init(&self) {}
}

#[derive(CubeType)]
pub struct GlobalToTileReader {}

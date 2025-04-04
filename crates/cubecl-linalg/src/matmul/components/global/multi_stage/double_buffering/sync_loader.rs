use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::Ident;
use crate::matmul::components::global::CommonGlobalConfig;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{
    BufferLoader, SyncBufferLoaderTrait, SyncBufferLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::single_buffer::BufferReader;
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(Clone, CubeType)]
pub struct SyncBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<EG, ES, CommonGlobalConfig<S>> for SyncBufferLoader<EG, ES, S, L>
{
    type StageReader = BufferReader<ES, L::TilingLayout>;

    fn reader(this: &Self, #[comptime] buffer_id: BufferId) -> Self::StageReader {
        BufferReader::new(this.stage, buffer_id, this.ident)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoaderTrait<EG, ES, CommonGlobalConfig<S>> for SyncBufferLoader<EG, ES, S, L>
{
    fn fill_stage(
        this: &mut Self,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EG, ES, CommonGlobalConfig<S>>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_index(),
            this.ident,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<EG, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: Ident,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(ident, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData::<S>,
        }
    }
}

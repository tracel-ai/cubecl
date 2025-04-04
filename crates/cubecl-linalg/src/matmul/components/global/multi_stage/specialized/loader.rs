use std::marker::PhantomData;

use crate::matmul::components::Ident;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::SyncBufferLoadingStrategy;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::single_buffer::BufferReader;
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use super::config::Config;

#[derive(CubeType)]
pub struct SyncBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    is_producer: bool,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _config: PhantomData<S>,
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
        is_producer: bool,
        #[comptime] ident: Ident,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(ident, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            is_producer,
            ident,
            _config: PhantomData,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] buffer_id: BufferId,
    ) -> BufferReader<ES, <L as SyncBufferLoadingStrategy>::TilingLayout> {
        BufferReader::new(this.stage, buffer_id, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] buffer_id: BufferId, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<EG, ES, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                buffer_id.to_index(),
                this.ident,
                config,
            );
        }
    }
}

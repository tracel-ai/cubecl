use std::marker::PhantomData;

use crate::matmul::components::Ident;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::multi_stage::{
    BufferLoader, SyncBufferLoader, SyncBufferLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use super::config::Config;

#[derive(CubeType)]
pub struct SyncLhsBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    is_producer: bool,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct SyncRhsBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    is_producer: bool,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<EG, ES, Config<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    type StageReader = LhsBufferReader<ES, L::TilingLayout>;

    fn reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        LhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<EG, ES, Config<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] buffer_id: BufferId, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<EG, ES, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                buffer_id.to_u32(),
                this.scaling,
                Ident::Lhs,
                config,
            );
        }
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncLhsBufferLoader<EI, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        scaling: CubeOption<ES>,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            is_producer,
            scaling,
            _config: PhantomData,
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<EG, ES, Config<S>> for SyncRhsBufferLoader<EG, ES, S, L>
{
    type StageReader = RhsBufferReader<ES, L::TilingLayout>;

    fn reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        RhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<EI, ES, Config<S>> for SyncRhsBufferLoader<EI, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<EI, ES, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                buffer.to_u32(),
                this.scaling,
                Ident::Rhs,
                config,
            );
        }
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncRhsBufferLoader<EI, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        scaling: CubeOption<ES>,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            is_producer,
            scaling,
            _config: PhantomData::<S>,
        }
    }
}

use std::marker::PhantomData;

use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::loader::sync::SyncBufferLoadingStrategy;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{
    CommonGlobalConfig, InputBufferLoader, SyncInputBufferLoader,
};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(CubeType)]
pub struct SyncLhsBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    num_buffers: u32,
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
    buffer_iter: u32,
    num_buffers: u32,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    InputBufferLoader<EG, ES, CommonGlobalConfig<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    type StageReader = LhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        LhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncInputBufferLoader<EG, ES, CommonGlobalConfig<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage(
        this: &mut Self,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EG, ES, CommonGlobalConfig<S>>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            Ident::Lhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncLhsBufferLoader<EG, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            num_buffers: config.tiling_dimensions(Ident::Lhs).tile_count_col(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    InputBufferLoader<EG, ES, CommonGlobalConfig<S>> for SyncRhsBufferLoader<EG, ES, S, L>
{
    type StageReader = RhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        RhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncInputBufferLoader<EG, ES, CommonGlobalConfig<S>> for SyncRhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage(
        this: &mut Self,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EG, ES, CommonGlobalConfig<S>>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            Ident::Rhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncRhsBufferLoader<EG, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Rhs).tile_count_row(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

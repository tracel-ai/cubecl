use std::marker::PhantomData;

use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::loader::sync::SyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{InputLoader, SyncInputLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
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
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
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
    is_producer: bool,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    InputLoader<EG, ES, Config<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    type StageReader = LhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsBufferReader::new(this.stage, this.buffer_iter)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncInputLoader<EG, ES, Config<S>> for SyncLhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<EG, ES, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                this.buffer_iter,
                Ident::Lhs,
                config,
            );
        }
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
        is_producer: bool,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Lhs).tile_count_col(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    InputLoader<EG, ES, Config<S>> for SyncRhsBufferLoader<EG, ES, S, L>
{
    type StageReader = RhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        RhsBufferReader::new(this.stage, this.buffer_iter)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: Config<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncInputLoader<EG, ES, Config<S>> for SyncRhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<EG, ES, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                this.buffer_iter,
                Ident::Rhs,
                config,
            );
        }
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
        is_producer: bool,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Rhs).tile_count_row(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

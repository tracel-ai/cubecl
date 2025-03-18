use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{AsyncBufferLoader, BufferLoader};
use crate::matmul::components::global::single_stage::AsyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CommonGlobalConfig, CopyMechanism};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(CubeType)]
pub struct AsyncLhsBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    num_buffers: u32,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct AsyncRhsBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    buffer_iter: u32,
    num_buffers: u32,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<EG, ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<EG, ES, S, L>
{
    type StageReader = LhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        LhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<EG, ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EG, ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            mechanism,
            Ident::Lhs,
            config,
        );
    }

    fn clear_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncLhsBufferLoader<EG, ES, S, L>
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

        AsyncLhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            num_buffers: config.tiling_dimensions(Ident::Lhs).tile_count_col(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<EG, ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<EG, ES, S, L>
{
    type StageReader = RhsBufferReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        RhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<EG, ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<EG, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EG, ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            mechanism,
            Ident::Rhs,
            config,
        );
    }

    fn clear_stage(this: &mut Self, #[comptime] config: CommonGlobalConfig<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncRhsBufferLoader<EG, ES, S, L>
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

        AsyncRhsBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Rhs).tile_count_row(),
            _config: PhantomData::<S>.runtime(),
        }
    }
}

use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::Ident;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{AsyncBufferLoader, BufferLoader};
use crate::matmul::components::global::single_stage::AsyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CommonGlobalConfig, CopyMechanism};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::CubeOption;

#[derive(CubeType)]
pub struct AsyncLhsBufferLoader<
    EI: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EI>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct AsyncRhsBufferLoader<
    EI: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EI>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<EI, ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<EI, ES, S, L>
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
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<EI, ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<EI, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EI, ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            this.scaling,
            buffer.to_u32(),
            Ident::Lhs,
            config,
        );
    }

    fn clear_stage(
        this: &mut Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        this.stage
            .clear_buffer::<S>(buffer_id, Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncLhsBufferLoader<EI, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        scaling: CubeOption<ES>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            scaling,
            _config: PhantomData::<S>,
        }
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<EI, ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<EI, ES, S, L>
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
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<EI, ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<EI, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EI, ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            this.scaling,
            buffer.to_u32(),
            Ident::Rhs,
            config,
        );
    }

    fn clear_stage(
        this: &mut Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        this.stage
            .clear_buffer::<S>(buffer_id, Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncRhsBufferLoader<EI, ES, S, L>
{
    pub fn new(
        tensor: VirtualTensor<EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        scaling: CubeOption<ES>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncRhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            scaling,
            _config: PhantomData::<S>,
        }
    }
}

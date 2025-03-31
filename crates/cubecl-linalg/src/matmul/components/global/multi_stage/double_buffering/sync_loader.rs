use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::Ident;
use crate::matmul::components::global::CommonGlobalConfig;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{
    BufferLoader, SyncBufferLoader, SyncBufferLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::CubeOption;

#[derive(Clone, CubeType)]
pub struct SyncLhsBufferLoader<
    EI: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EI>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(Clone, CubeType)]
pub struct SyncRhsBufferLoader<
    EI: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EI>,
    pub stage: Stage<ES, L::TilingLayout>,
    pub scaling: CubeOption<ES>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<EI, ES, CommonGlobalConfig<S>> for SyncLhsBufferLoader<EI, ES, S, L>
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
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<EI, ES, CommonGlobalConfig<S>> for SyncLhsBufferLoader<EI, ES, S, L>
{
    fn fill_stage(
        this: &mut Self,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EI, ES, CommonGlobalConfig<S>>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            this.scaling,
            Ident::Lhs,
            config,
        );
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
        scaling: CubeOption<ES>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            scaling,
            _config: PhantomData::<S>,
        }
    }
}

#[cube]
impl<EI: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<EI, ES, CommonGlobalConfig<S>> for SyncRhsBufferLoader<EI, ES, S, L>
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
    SyncBufferLoader<EI, ES, CommonGlobalConfig<S>> for SyncRhsBufferLoader<EI, ES, S, L>
{
    fn fill_stage(
        this: &mut Self,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<EI, ES, CommonGlobalConfig<S>>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_u32(),
            this.scaling,
            Ident::Rhs,
            config,
        );
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
        scaling: CubeOption<ES>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsBufferLoader::<EI, ES, S, L> {
            tensor_view,
            stage,
            scaling,
            _config: PhantomData::<S>,
        }
    }
}

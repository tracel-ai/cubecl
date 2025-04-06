use std::marker::PhantomData;

use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::multi_stage::{
    BufferLoader, SyncBufferLoader, SyncBufferLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{Ident, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use super::config::Config;

#[derive(CubeType)]
pub struct SyncLhsBufferLoader<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    is_producer: bool,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct SyncRhsBufferLoader<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    L: SyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    is_producer: bool,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<MP::EI, MP::ES, Config<S>> for SyncLhsBufferLoader<MP, S, L>
{
    type StageReader = LhsBufferReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        LhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<MP::EI, MP::ES, Config<S>> for SyncLhsBufferLoader<MP, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] buffer_id: BufferId, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<MP, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                buffer_id.to_u32(),
                this.quantization,
                Ident::Lhs,
                config,
            );
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncLhsBufferLoader<MP, S, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsBufferLoader::<MP, S, L> {
            tensor_view,
            stage,
            is_producer,
            quantization,
            _config: PhantomData,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    BufferLoader<MP::EI, MP::ES, Config<S>> for SyncRhsBufferLoader<MP, S, L>
{
    type StageReader = RhsBufferReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self, #[comptime] buffer: BufferId) -> Self::StageReader {
        RhsBufferReader::new(this.stage, buffer.to_u32())
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<MP::EI, MP::ES, Config<S>> for SyncRhsBufferLoader<MP, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: Config<S>) {
        if this.is_producer {
            L::load_buffer::<MP, Config<S>>(
                &this.tensor_view,
                &mut this.stage,
                buffer.to_u32(),
                this.quantization,
                Ident::Rhs,
                config,
            );
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncBufferLoadingStrategy>
    SyncRhsBufferLoader<MP, S, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsBufferLoader::<MP, S, L> {
            tensor_view,
            stage,
            quantization,
            is_producer,
            _config: PhantomData::<S>,
        }
    }
}

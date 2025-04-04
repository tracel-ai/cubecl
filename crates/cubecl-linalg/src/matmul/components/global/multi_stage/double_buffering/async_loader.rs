use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{AsyncBufferLoader, BufferLoader};
use crate::matmul::components::global::single_stage::AsyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CommonGlobalConfig, CopyMechanism, Quantization};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{Ident, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(CubeType)]
pub struct AsyncLhsBufferLoader<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct AsyncRhsBufferLoader<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<MP::EI, MP::ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<MP, S, L>
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
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<MP::EI, MP::ES, CommonGlobalConfig<S>> for AsyncLhsBufferLoader<MP, S, L>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<MP::EI, MP::ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
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
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncLhsBufferLoader<MP, S, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLhsBufferLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<MP::EI, MP::ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<MP, S, L>
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
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<MP::EI, MP::ES, CommonGlobalConfig<S>> for AsyncRhsBufferLoader<MP, S, L>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<MP::EI, MP::ES, CommonGlobalConfig<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
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
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncRhsBufferLoader<MP, S, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncRhsBufferLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

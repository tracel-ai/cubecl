use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::Ident;
use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::{AsyncBufferLoaderTrait, BufferLoader};
use crate::matmul::components::global::single_stage::AsyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CommonGlobalConfig, CopyMechanism};
use crate::matmul::components::stage::single_buffer::BufferReader;
use crate::matmul::components::stage::{self, Stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(CubeType)]
pub struct AsyncBufferLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: AsyncBufferLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    BufferLoader<EG, ES, CommonGlobalConfig<S>> for AsyncBufferLoader<EG, ES, S, L>
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
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoaderTrait<EG, ES, CommonGlobalConfig<S>> for AsyncBufferLoader<EG, ES, S, L>
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
            mechanism,
            buffer.to_index(),
            this.ident,
            config,
        );
    }

    fn clear_stage(
        this: &mut Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        this.stage
            .clear_buffer::<S>(buffer_id, this.ident, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncBufferLoadingStrategy>
    AsyncBufferLoader<EG, ES, S, L>
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

        AsyncBufferLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData::<S>,
        }
    }
}

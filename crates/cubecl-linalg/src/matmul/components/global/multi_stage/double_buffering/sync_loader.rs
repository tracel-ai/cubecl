use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::Ident;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::multi_stage::SyncBufferLoadingStrategy;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::single_buffer::BufferReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(Clone, CubeType)]
pub struct SyncBufferLoader<EG: Numeric, ES: Numeric, G: GlobalConfig, L: SyncBufferLoadingStrategy>
{
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: GlobalConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<EG, ES, G, L>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(ident, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncBufferLoader::<EG, ES, G, L> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] buffer_id: BufferId,
    ) -> BufferReader<ES, L::TilingLayout> {
        BufferReader::new(this.stage, buffer_id, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] buffer: BufferId, #[comptime] config: G) {
        L::load_buffer::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage,
            buffer.to_index(),
            this.ident,
            config,
        );
    }
}

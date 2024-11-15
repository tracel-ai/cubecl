use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::Stage;
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsBufferLoader<EG, ES> {
    type StageReader = LhsBufferReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        if this.is_producer {
            // TODO load if producer
            // BufferLoading::load_to_slice::<EG, ES, G>(
            //     &this.tensor_view,
            //     &mut this.stage.as_slice_mut(),
            //     Ident::Lhs,
            //     config,
            // );
        }

        LhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> LhsBufferLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        is_producer: bool,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, nth_batch);

        LhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_iter: 0,
            num_buffers: config.stage_dim(Ident::Lhs).num_tiles_y,
            is_producer,
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for RhsBufferLoader<EG, ES> {
    type StageReader = RhsBufferReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] _config: G) -> Self::StageReader {
        if this.is_producer {
            // TODO load if producer
        }

        RhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> RhsBufferLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        is_producer: bool,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, nth_batch);

        RhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_iter: 0,
            num_buffers: config.stage_dim(Ident::Lhs).num_tiles_y,
            is_producer,
        }
    }
}

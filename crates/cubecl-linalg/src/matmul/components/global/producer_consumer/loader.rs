use crate::matmul::components::global::producer_consumer::buffer_loading::BufferLoading;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Config;
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

    fn fill_stage<G: Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        if this.is_producer {
            load_buffer::<EG, ES, G>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Lhs,
                config,
            );
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

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        if this.is_producer {
            load_buffer::<EG, ES, G>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Rhs,
                config,
            );
        }

        RhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Rhs);
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
            num_buffers: config.stage_dim(Ident::Rhs).num_tiles_x,
            is_producer,
        }
    }
}

#[cube]
fn load_buffer<EG: Numeric, ES: Numeric, G: global::Config>(
    buffer_iter: u32,
    tensor_view: &TensorReader<EG>,
    stage: &mut Stage<ES>,
    #[comptime] ident: Ident,
    #[comptime] config: G,
) {
    let tile_num_elements = config.stage_dim(ident).tile_num_elements();
    let line_size = config.stage_line_size(ident);
    let start = buffer_iter * tile_num_elements / line_size;
    let end = start + tile_num_elements / line_size;

    BufferLoading::load_to_slice::<EG, ES, G>(
        &tensor_view,
        &mut stage.as_slice_mut().slice_mut(start, end),
        ident,
        config,
    );
}

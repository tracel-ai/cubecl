use std::marker::PhantomData;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global::base::Config as _;
use crate::matmul::components::global::buffered::buffer_loading::BufferLoading;
use crate::matmul::components::global::buffered::pipelined;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{Loader, LoadingStrategy};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_id: u32,
    num_buffers: u32,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_id: u32,
    num_buffers: u32,
}

#[cube]
impl<EG: Numeric, ES: Numeric> LhsBufferLoader<EG, ES> {
    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        buffer_id: u32,
        #[comptime] config: pipelined::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_id,
            num_buffers: config.stage_dim(Ident::Lhs).num_tiles_y_dim(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsBufferLoader<EG, ES> {
    type StageReader = LhsBufferReader<ES>;
    type LoadBuffer = <BufferLoading as LoadingStrategy<EG, ES>>::LoadBuffer;

    fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> Self::LoadBuffer {
        todo!()
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: Self::LoadBuffer,
        #[comptime] config: G,
    ) -> Self::StageReader {
        todo!()
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        todo!()
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for RhsBufferLoader<EG, ES> {
    type StageReader = RhsBufferReader<ES>;
    type LoadBuffer = <BufferLoading as LoadingStrategy<EG, ES>>::LoadBuffer;

    fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> Self::LoadBuffer {
        todo!()
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: Self::LoadBuffer,
        #[comptime] config: G,
    ) -> Self::StageReader {
        todo!()
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        todo!()
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> RhsBufferLoader<EG, ES> {
    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        buffer_id: u32,
        #[comptime] config: pipelined::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_id,
            num_buffers: config.stage_dim(Ident::Rhs).num_tiles_x_dim(),
        }
    }
}

#[cube]
fn load_buffer<EG: Numeric, ES: Numeric, S: stage::Config>(
    buffer_iter: u32,
    tensor_view: &TensorReader<EG>,
    stage: &mut Stage<ES>,
    #[comptime] ident: Ident,
    #[comptime] config: pipelined::Config<S>,
) {
    let buffer_num_elements = config.stage_dim(ident).buffer_num_elements();
    let line_size = config.stage_line_size(ident);
    let buffer_num_lines = buffer_num_elements / line_size;

    #[allow(clippy::all)]
    let _ = comptime!(check_buffers_contiguous(ident, config));

    let start = buffer_iter * buffer_num_lines;
    let end = start + buffer_num_lines;
    let buffer_slice = &mut stage.as_slice_mut().slice_mut(start, end);

    BufferLoading::load_to_slice::<EG, ES, pipelined::Config<S>>(
        tensor_view,
        buffer_slice,
        config.num_planes(),
        0u32,
        ident,
        config,
    );
}

fn check_buffers_contiguous<G: global::Config>(ident: Ident, config: G) {
    match ident.as_input() {
        InputIdent::Lhs => {
            if let TilingOrderConfig::RowMajor = config.tiling_order(ident) {
                panic!("Lhs must have ColMajor tiling order in pipelined setting")
            }
        }
        InputIdent::Rhs => {
            if let TilingOrderConfig::ColMajor = config.tiling_order(ident) {
                panic!("Rhs must have RowMajor tiling order in pipelined setting")
            }
        }
    }
}

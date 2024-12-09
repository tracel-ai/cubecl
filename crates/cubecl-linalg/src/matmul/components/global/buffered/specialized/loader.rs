use std::marker::PhantomData;

use crate::matmul::components::config::InputIdent;
use crate::matmul::components::global::args::{GmmArgs, TensorInput};
use crate::matmul::components::global::base::Config as _;
use crate::matmul::components::global::buffered::buffer_loading::BufferLoading;
use crate::matmul::components::global::buffered::specialized;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsBufferLoader<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config> {
    pub tensor_view: TensorReader<GA, EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config> {
    pub tensor_view: TensorReader<GA, EG>,
    pub stage: Stage<ES>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
    _config: PhantomData<S>,
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config>
    Loader<EG, ES, specialized::Config<S>> for LhsBufferLoader<GA, EG, ES, S>
{
    type StageReader = LhsBufferReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: specialized::Config<S>) {
        if this.is_producer {
            load_buffer::<GA, EG, ES, S>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Lhs,
                config,
            );
        }
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
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
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config> LhsBufferLoader<GA, EG, ES, S> {
    pub fn new(
        tensor: TensorInput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        #[comptime] config: specialized::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsBufferLoader::<GA, EG, ES, S> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.stage_dim(Ident::Lhs).num_tiles_y_dim(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config>
    Loader<EG, ES, specialized::Config<S>> for RhsBufferLoader<GA, EG, ES, S>
{
    type StageReader = RhsBufferReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: specialized::Config<S>) {
        if this.is_producer {
            load_buffer::<GA, EG, ES, S>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Rhs,
                config,
            );
        }
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
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
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config> RhsBufferLoader<GA, EG, ES, S> {
    pub fn new(
        tensor: TensorInput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        #[comptime] config: specialized::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsBufferLoader::<GA, EG, ES, S> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.stage_dim(Ident::Rhs).num_tiles_x_dim(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
fn load_buffer<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config>(
    buffer_iter: u32,
    tensor_view: &TensorReader<GA, EG>,
    stage: &mut Stage<ES>,
    #[comptime] ident: Ident,
    #[comptime] config: specialized::Config<S>,
) {
    let buffer_num_elements = config.stage_dim(ident).buffer_num_elements();
    let line_size = config.stage_line_size(ident);
    let buffer_num_lines = buffer_num_elements / line_size;

    #[allow(clippy::all)]
    let _ = comptime!(check_buffers_contiguous(ident, config));

    let start = buffer_iter * buffer_num_lines;
    let end = start + buffer_num_lines;
    let buffer_slice = &mut stage.as_slice_mut().slice_mut(start, end);

    BufferLoading::load_to_slice::<GA, EG, ES, specialized::Config<S>>(
        tensor_view,
        buffer_slice,
        config.num_producers(),
        config.num_consumers(),
        ident,
        config,
    );
}

fn check_buffers_contiguous<G: global::Config>(ident: Ident, config: G) {
    match ident.as_input() {
        InputIdent::Lhs => {
            if let TilingOrderConfig::RowMajor = config.tiling_order(ident) {
                panic!("Lhs must have ColMajor tiling order in producer consumer setting")
            }
        }
        InputIdent::Rhs => {
            if let TilingOrderConfig::ColMajor = config.tiling_order(ident) {
                panic!("Rhs must have RowMajor tiling order in producer consumer setting")
            }
        }
    }
}

use std::marker::PhantomData;

use crate::matmul::components::global::base::GlobalConfig as _;
use crate::matmul::components::global::multi_stage::buffer_loading::BufferLoading;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{InputLoader, SyncInputLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use super::config::Config;

#[derive(CubeType)]
pub struct LhsBufferLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, T>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, T>,
    buffer_iter: u32,
    num_buffers: u32,
    is_producer: bool,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    InputLoader<EG, ES, Config<S>> for LhsBufferLoader<EG, ES, S, T>
{
    type StageReader = LhsBufferReader<ES, T>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsBufferReader::<ES, T> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    SyncInputLoader<EG, ES, Config<S>> for LhsBufferLoader<EG, ES, S, T>
{
    fn fill_stage(this: &mut Self, #[comptime] config: Config<S>) {
        if this.is_producer {
            load_buffer::<EG, ES, S, T>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Lhs,
                config,
            );
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    LhsBufferLoader<EG, ES, S, T>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsBufferLoader::<EG, ES, S, T> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Lhs).tile_count_col(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    InputLoader<EG, ES, Config<S>> for RhsBufferLoader<EG, ES, S, T>
{
    type StageReader = RhsBufferReader<ES, T>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        RhsBufferReader::<ES, T> {
            stage: this.stage,
            buffer: this.buffer_iter,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.buffer_iter = (this.buffer_iter + 1) % this.num_buffers;
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: Config<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    SyncInputLoader<EG, ES, Config<S>> for RhsBufferLoader<EG, ES, S, T>
{
    fn fill_stage(this: &mut Self, #[comptime] config: Config<S>) {
        if this.is_producer {
            load_buffer::<EG, ES, S, T>(
                this.buffer_iter,
                &this.tensor_view,
                &mut this.stage,
                Ident::Rhs,
                config,
            );
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>
    RhsBufferLoader<EG, ES, S, T>
{
    pub fn new(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        is_producer: bool,
        #[comptime] config: Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsBufferLoader::<EG, ES, S, T> {
            tensor_view,
            stage,
            buffer_iter: 0u32.runtime(),
            num_buffers: config.tiling_dimensions(Ident::Rhs).tile_count_row(),
            is_producer,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
fn load_buffer<EG: Numeric, ES: Numeric, S: stage::StageConfig, T: TilingLayout>(
    buffer_iter: u32,
    tensor_view: &TensorReader<EG>,
    stage: &mut Stage<ES, T>,
    #[comptime] ident: Ident,
    #[comptime] config: Config<S>,
) {
    let buffer_num_elements = config
        .tiling_dimensions(ident)
        .buffer_size(ident.as_input());
    let line_size = config.stage_line_size(ident);
    let buffer_num_lines = buffer_num_elements / line_size;

    let start = buffer_iter * buffer_num_lines;
    let end = start + buffer_num_lines;
    let buffer_slice = &mut stage.as_slice_mut().slice_mut(start, end);

    BufferLoading::load_to_slice::<EG, ES, Config<S>>(
        tensor_view,
        buffer_slice,
        config.num_producers(),
        config.num_consumers(),
        ident,
        config,
    );
}

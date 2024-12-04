use crate::matmul::components::global::base::Config as _;
use crate::matmul::components::global::buffered::buffer_loading::{buffer_slice, BufferLoading};
use crate::matmul::components::global::buffered::pipelined;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{LoadBuffer, Loader, LoadingStrategy};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_id: u32,
}

#[derive(CubeType)]
pub struct RhsBufferLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    buffer_id: u32,
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
        let buffer_offset = buffer_id * config.stage_dim(Ident::Rhs).tile_size_x_dim();
        let tensor_view =
            TensorReader::new(tensor, x_offset, y_offset + buffer_offset, batch_offset);

        LhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_id,
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsBufferLoader<EG, ES> {
    type StageReader = LhsBufferReader<ES>;

    fn init_buffer<G: global::Config>(#[comptime] config: G) -> LoadBuffer<EG> {
        BufferLoading::<EG, ES>::init_buffer::<G>(Ident::Lhs, config)
    }

    fn fetch_global<G: global::Config>(
        this: &Self,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) {
        BufferLoading::<EG, ES>::fetch::<G>(&this.tensor_view, buffer, Ident::Lhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: &SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        BufferLoading::store::<G>(
            buffer,
            &mut buffer_slice::<EG, ES, G>(this.buffer_id, &mut this.stage, Ident::Lhs, config),
            Ident::Lhs,
            config,
        );
        LhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_id,
        }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Lhs;
        let k_offset = config.stage_dim(ident).num_elements_y_dim();
        this.tensor_view.update_view(k_offset, ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for RhsBufferLoader<EG, ES> {
    type StageReader = RhsBufferReader<ES>;

    fn init_buffer<G: global::Config>(#[comptime] config: G) -> LoadBuffer<EG> {
        BufferLoading::<EG, ES>::init_buffer::<G>(Ident::Rhs, config)
    }

    fn fetch_global<G: global::Config>(
        this: &Self,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) {
        BufferLoading::<EG, ES>::fetch::<G>(&this.tensor_view, buffer, Ident::Rhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: &SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        BufferLoading::store::<G>(
            buffer,
            &mut buffer_slice::<EG, ES, G>(this.buffer_id, &mut this.stage, Ident::Rhs, config),
            Ident::Rhs,
            config,
        );
        RhsBufferReader::<ES> {
            stage: this.stage,
            buffer: this.buffer_id,
        }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Rhs;
        let k_offset = config.stage_dim(ident).num_elements_x_dim();
        this.tensor_view.update_view(k_offset, ident);
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
        let buffer_offset = buffer_id * config.stage_dim(Ident::Rhs).tile_size_x_dim();
        let tensor_view =
            TensorReader::new(tensor, x_offset + buffer_offset, y_offset, batch_offset);

        RhsBufferLoader::<EG, ES> {
            tensor_view,
            stage,
            buffer_id,
        }
    }
}

#[cube]
// TODO refactor, it's a bit hacky
impl<EG: Numeric, ES: Numeric, L: Loader<EG, ES>> Loader<EG, ES> for (L, L) {
    type StageReader = <L as Loader<EG, ES>>::StageReader;

    fn init_buffer<G: global::Config>(#[comptime] config: G) -> LoadBuffer<EG> {
        let _ = comptime!(unavailable_method());

        // Just to make the compiler happy
        L::init_buffer::<G>(config)
    }

    fn fetch_global<G: global::Config>(
        this: &Self,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) {
        let _ = comptime!(unavailable_method());

        // Just to make the compiler happy
        L::fetch_global::<G>(&this.0, buffer, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: &SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        let _ = comptime!(unavailable_method());

        // Just to make the compiler happy
        L::fill_stage::<G>(&mut this.0, buffer, config)
    }

    fn to_next_stage<G: global::Config>(_this: &mut Self, #[comptime] _config: G) {
        let _ = comptime!(unavailable_method());
    }
}

fn unavailable_method() {
    panic!("Cannot call method directly on loader pair. Try calling it on underlying loaders.");
}

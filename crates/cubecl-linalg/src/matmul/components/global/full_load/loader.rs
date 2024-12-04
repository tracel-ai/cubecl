use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{Config, LoadBuffer, Loader, LoadingStrategy};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::Stage;
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> Loader<EG, ES> for LhsLoader<EG, ES, L> {
    type StageReader = LhsReader<ES>;

    fn init_buffer<G: Config>(#[comptime] config: G) -> LoadBuffer<EG> {
        L::init_buffer::<G>(Ident::Lhs, config)
    }

    fn fetch_global<G: global::Config>(
        this: &Self,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) {
        L::fetch::<G>(&this.tensor_view, buffer, Ident::Lhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: &Slice<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        L::store::<G>(buffer, &mut this.stage.as_slice_mut(), Ident::Lhs, config);
        LhsReader::<ES> { stage: this.stage }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Lhs;
        let k_offset = config.stage_dim(ident).num_elements_y_dim();
        this.tensor_view.update_view(k_offset, ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> LhsLoader<EG, ES, L> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> Loader<EG, ES> for RhsLoader<EG, ES, L> {
    type StageReader = RhsReader<ES>;

    fn init_buffer<G: Config>(#[comptime] config: G) -> LoadBuffer<EG> {
        L::init_buffer::<G>(Ident::Rhs, config)
    }

    fn fetch_global<G: global::Config>(
        this: &Self,
        buffer: &mut SliceMut<Line<EG>>,
        #[comptime] config: G,
    ) {
        L::fetch::<G>(&this.tensor_view, buffer, Ident::Rhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: &Slice<Line<EG>>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        L::store::<G>(buffer, &mut this.stage.as_slice_mut(), Ident::Rhs, config);
        RhsReader::<ES> { stage: this.stage }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Rhs;
        let k_offset = config.stage_dim(ident).num_elements_x_dim();
        this.tensor_view.update_view(k_offset, ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> RhsLoader<EG, ES, L> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

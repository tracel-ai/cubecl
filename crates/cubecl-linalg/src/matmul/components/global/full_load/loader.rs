use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::LoadBuffer;
use crate::matmul::components::global::{full_load, Config};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> LhsLoader<EG, ES, L> {
    pub fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> LoadBuffer<EG> {
        L::fetch::<EG, G>(&this.tensor_view, Ident::Lhs, config)
    }

    pub fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: LoadBuffer<EG>,
        #[comptime] config: G,
    ) -> LhsReader<ES> {
        L::store::<EG, ES, G>(buffer, &mut this.stage.as_slice_mut(), Ident::Lhs, config);
        LhsReader::<ES> { stage: this.stage }
    }

    pub fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Lhs;
        let k_offset = config.stage_dim(ident).num_elements_y_dim();
        this.tensor_view.update_view(k_offset, ident);
    }

    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: full_load::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> RhsLoader<EG, ES, L> {
    pub fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> LoadBuffer<EG> {
        L::fetch::<EG, G>(&this.tensor_view, Ident::Rhs, config)
    }

    pub fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: LoadBuffer<EG>,
        #[comptime] config: G,
    ) -> RhsReader<ES> {
        // TODO RhsReader should be rhs buffer reader for buffered
        L::store::<EG, ES, G>(buffer, &mut this.stage.as_slice_mut(), Ident::Rhs, config);
        RhsReader::<ES> { stage: this.stage }
    }

    pub fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Rhs;
        let k_offset = config.stage_dim(ident).num_elements_x_dim();
        this.tensor_view.update_view(k_offset, ident);
    }

    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: full_load::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
pub trait LoadingStrategy: 'static + Send + Sync + Clone {
    fn fetch<EG: Numeric, G: global::Config>(
        read_view: &TensorReader<EG>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> LoadBuffer<EG>;

    fn store<EG: Numeric, ES: Numeric, G: global::Config>(
        load_buffer: LoadBuffer<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

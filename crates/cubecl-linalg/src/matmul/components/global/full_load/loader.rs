use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Loader;
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
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> Loader<EG, ES> for LhsLoader<EG, ES, L> {
    type StageReader = LhsReader<ES>;
    type Config<S: stage::Config> = full_load::Config<S>;

    fn fill_stage<S: stage::Config>(this: &mut Self, #[comptime] config: Self::Config<S>) {
        L::load_to_slice::<EG, ES, Self::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> LhsLoader<EG, ES, L> {
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
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> Loader<EG, ES> for RhsLoader<EG, ES, L> {
    type StageReader = RhsReader<ES>;
    type Config<S: stage::Config> = full_load::Config<S>;

    fn fill_stage<S: stage::Config>(this: &mut Self, #[comptime] config: full_load::Config<S>) {
        L::load_to_slice::<EG, ES, full_load::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy> RhsLoader<EG, ES, L> {
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
    fn load_to_slice<EG: Numeric, ES: Numeric, G: global::Config>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

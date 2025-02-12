use std::marker::PhantomData;

use crate::matmul::components::global::single_stage;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{InputLoader, LoadingValidation};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use crate::tensor::VirtualTensor;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy>
    InputLoader<EG, ES, single_stage::Config<S>> for LhsLoader<EG, ES, S, L>
{
    type StageReader = LhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_to_slice::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
    }

    fn fill_stage_window(
        this: &mut Self,
        pipeline: Pipeline<ES>,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_window::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            pipeline,
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
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy> LhsLoader<EG, ES, S, L> {
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy>
    InputLoader<EG, ES, single_stage::Config<S>> for RhsLoader<EG, ES, S, L>
{
    type StageReader = RhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_to_slice::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
    }

    fn fill_stage_window(
        this: &mut Self,
        pipeline: Pipeline<ES>,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_window::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            pipeline,
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
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: LoadingStrategy> RhsLoader<EG, ES, S, L> {
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
pub trait LoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    fn load_to_slice<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    fn load_window<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        pipeline: Pipeline<ES>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

use std::marker::PhantomData;

use crate::matmul::components::global::homogeneous;
use crate::matmul::components::global::homogeneous::cyclic_loading::CyclicLoading;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, S: stage::Config> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, S: stage::Config> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::Config> Loader<EG, ES, homogeneous::Config<S>>
    for LhsLoader<EG, ES, S>
{
    type StageReader = LhsReader<ES>;

    fn fill_stage(
        this: &mut Self,
        #[comptime] config: homogeneous::Config<S>,
    ) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, homogeneous::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::Config> LhsLoader<EG, ES, S> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<EG, ES, S> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::Config> Loader<EG, ES, homogeneous::Config<S>>
    for RhsLoader<EG, ES, S>
{
    type StageReader = RhsReader<ES>;

    fn fill_stage(
        this: &mut Self,
        #[comptime] config: homogeneous::Config<S>,
    ) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, homogeneous::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::Config> RhsLoader<EG, ES, S> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<EG, ES, S> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

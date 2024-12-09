use std::marker::PhantomData;

use crate::matmul::components::global::args::{GmmArgs, TensorInput};
use crate::matmul::components::global::full_load;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<
    GA: GmmArgs<EG>,
    EG: Numeric,
    ES: Numeric,
    S: stage::Config,
    L: LoadingStrategy,
> {
    pub tensor_view: TensorReader<GA, EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct RhsLoader<
    GA: GmmArgs<EG>,
    EG: Numeric,
    ES: Numeric,
    S: stage::Config,
    L: LoadingStrategy,
> {
    pub tensor_view: TensorReader<GA, EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config, L: LoadingStrategy>
    Loader<EG, ES, full_load::Config<S>> for LhsLoader<GA, EG, ES, S, L>
{
    type StageReader = LhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: full_load::Config<S>) {
        L::load_to_slice::<GA, EG, ES, full_load::Config<S>>(
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
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config, L: LoadingStrategy>
    LhsLoader<GA, EG, ES, S, L>
{
    pub fn new<G: global::Config>(
        tensor: TensorInput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<GA, EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config, L: LoadingStrategy>
    Loader<EG, ES, full_load::Config<S>> for RhsLoader<GA, EG, ES, S, L>
{
    type StageReader = RhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: full_load::Config<S>) {
        L::load_to_slice::<GA, EG, ES, full_load::Config<S>>(
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
impl<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, S: stage::Config, L: LoadingStrategy>
    RhsLoader<GA, EG, ES, S, L>
{
    pub fn new<G: global::Config>(
        tensor: TensorInput<EG, GA>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<GA, EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
pub trait LoadingStrategy: 'static + Send + Sync + Clone {
    fn load_to_slice<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, G: global::Config>(
        read_view: &TensorReader<GA, EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

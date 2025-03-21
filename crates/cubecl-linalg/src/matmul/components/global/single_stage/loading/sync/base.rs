use std::marker::PhantomData;

use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::single_stage;
use crate::matmul::components::global::single_stage::{FullLoader, SyncFullLoader};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::{Ident, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait SyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

#[derive(CubeType)]
pub struct SyncFullLhsLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncFullLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct SyncFullRhsLoader<
    EG: Numeric,
    ES: Numeric,
    S: stage::StageConfig,
    L: SyncFullLoadingStrategy,
> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    FullLoader<EG, ES, single_stage::Config<S>> for SyncFullLhsLoader<EG, ES, S, L>
{
    type StageReader = LhsReader<ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<EG, ES, single_stage::Config<S>> for SyncFullLhsLoader<EG, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage,
            Ident::Lhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLhsLoader<EG, ES, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncFullLhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    FullLoader<EG, ES, single_stage::Config<S>> for SyncFullRhsLoader<EG, ES, S, L>
{
    type StageReader = RhsReader<ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<EG, ES, single_stage::Config<S>> for SyncFullRhsLoader<EG, ES, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<EG, ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage,
            Ident::Rhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullRhsLoader<EG, ES, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncFullRhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

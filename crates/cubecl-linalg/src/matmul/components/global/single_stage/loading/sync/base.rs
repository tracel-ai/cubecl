use std::marker::PhantomData;

use crate::matmul::components::global::single_stage;
use crate::matmul::components::global::single_stage::{FullLoader, SyncFullLoader};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::multi_buffer::Reader;
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::{Ident, global};
use crate::matmul::components::{Lhs, MatmulPrecision};
use crate::matmul::components::{Rhs, global::LoadingValidation};
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
pub struct SyncFullLhsLoader<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
{
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct SyncFullRhsLoader<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
{
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    FullLoader<MP, single_stage::Config<S>> for SyncFullLhsLoader<MP, S, L>
{
    type StageReader = Reader<Lhs, MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        Reader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<MP, single_stage::Config<S>> for SyncFullLhsLoader<MP, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<MP::EI, MP::ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage,
            Ident::Lhs,
            config,
        );
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLhsLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncFullLhsLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    FullLoader<MP, single_stage::Config<S>> for SyncFullRhsLoader<MP, S, L>
{
    type StageReader = Reader<Rhs, MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        Reader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<MP, single_stage::Config<S>> for SyncFullRhsLoader<MP, S, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<MP::EI, MP::ES, single_stage::Config<S>>(
            &this.tensor_view,
            &mut this.stage,
            Ident::Rhs,
            config,
        );
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullRhsLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncFullRhsLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

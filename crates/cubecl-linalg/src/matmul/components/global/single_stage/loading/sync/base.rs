use std::marker::PhantomData;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::single_stage::{Loader, SyncLoader};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::{LhsReader, RhsReader};
use crate::matmul::components::stage::{Stage, TilingLayout};
use crate::matmul::components::{Ident, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait SyncLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

#[derive(CubeType, Clone)]
pub struct SyncLhsLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[derive(CubeType, Clone)]
pub struct SyncRhsLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> Loader<MP, G>
    for SyncLhsLoader<MP, G, L>
{
    type StageReader = LhsReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> SyncLoader<MP, G>
    for SyncLhsLoader<MP, G, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        L::load::<MP::EI, MP::ES, G>(&this.tensor_view, &mut this.stage, Ident::Lhs, config);
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> SyncLhsLoader<MP, G, L> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLhsLoader::<MP, G, L> {
            tensor_view,
            stage,
            _phantom: PhantomData::<(G, L)>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> Loader<MP, G>
    for SyncRhsLoader<MP, G, L>
{
    type StageReader = RhsReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> SyncLoader<MP, G>
    for SyncRhsLoader<MP, G, L>
{
    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        L::load::<MP::EI, MP::ES, G>(&this.tensor_view, &mut this.stage, Ident::Rhs, config);
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> SyncRhsLoader<MP, G, L> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncRhsLoader::<MP, G, L> {
            tensor_view,
            stage,
            _phantom: PhantomData::<(G, L)>,
        }
    }
}

use std::marker::PhantomData;

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::single_stage::Loader;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::StageReader;
use crate::matmul::components::stage::{Stage, TilingLayout};
use crate::matmul::components::{Ident, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait SyncLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage
    fn load<EG: Numeric, ES: Numeric, G: global::GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
}

#[derive(CubeType, Clone)]
pub struct SyncLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> Loader<MP, G>
    for SyncLoader<MP, G, L>
{
    type TilingLayout = L::TilingLayout;

    fn reader(this: &Self) -> StageReader<MP::ES, Self::TilingLayout> {
        StageReader::<MP::ES, Self::TilingLayout> {
            stage: this.stage,
            ident: comptime!(this.ident),
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncLoadingStrategy> SyncLoader<MP, G, L> {
    pub fn fill_stage(this: &mut Self, #[comptime] config: G) {
        L::load::<MP::EI, MP::ES, G>(&this.tensor_view, &mut this.stage, this.ident, config);
    }

    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(ident, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        SyncLoader::<MP, G, L> {
            tensor_view,
            stage,
            ident,
            _phantom: PhantomData::<(G, L)>,
        }
    }
}

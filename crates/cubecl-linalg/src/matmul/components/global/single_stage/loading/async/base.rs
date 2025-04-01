use std::marker::PhantomData;

use crate::matmul::components::global::single_stage::Loader;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{Stage, StageReader, TilingLayout};
use crate::matmul::components::{Ident, InputIdent, MatmulPrecision, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait AsyncLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage
    fn load<EI: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct AsyncLoader<MP: MatmulPrecision, G: GlobalConfig, L: AsyncLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: AsyncLoadingStrategy> AsyncLoader<MP, G, L> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self {
        let mut stage = Stage::new::<G::SmmConfig>(ident, config.to_smm_config());

        match ident.as_input() {
            InputIdent::Lhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_row_bounds(ident) {
                    if x_offset
                        > tensor.shape(tensor.rank() - 2)
                            - config.tiling_dimensions(ident).total_row()
                    {
                        stage.clear::<G::SmmConfig>(ident, config.to_smm_config());
                    }
                }
            }
            InputIdent::Rhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_col_bounds(ident) {
                    if y_offset
                        > tensor.shape(tensor.rank() - 1)
                            - config.tiling_dimensions(ident).total_col()
                    {
                        stage.clear::<G::SmmConfig>(ident, config.to_smm_config());
                    }
                }
            }
        }
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLoader::<MP, G, L> {
            tensor_view,
            stage,
            ident,
            _phantom: PhantomData::<(G, L)>,
        }
    }

    pub fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: G,
    ) {
        L::load::<MP::EI, MP::ES, G, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            this.ident,
            config,
        );
    }

    pub fn clear_stage(this: &mut Self, #[comptime] config: G) {
        this.stage
            .clear::<G::SmmConfig>(this.ident, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: AsyncLoadingStrategy> Loader<MP, G>
    for AsyncLoader<MP, G, L>
{
    type TilingLayout = L::TilingLayout;

    fn reader(this: &Self) -> StageReader<MP::ES, Self::TilingLayout> {
        StageReader::<MP::ES, Self::TilingLayout> {
            stage: this.stage,
            ident: Ident::Lhs,
        }
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

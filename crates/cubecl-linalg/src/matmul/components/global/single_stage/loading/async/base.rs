use std::marker::PhantomData;

use crate::matmul::components::global::single_stage;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::multi_buffer::FullReader;
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::{Ident, MatmulPrecision, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<EI: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[cube]
pub trait AsyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<EI: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EI>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct AsyncLoader<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L)>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    AsyncLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self {
        let mut stage = Stage::new::<G::SmmConfig>(ident, config.to_smm_config());

        #[allow(clippy::collapsible_if)]
        if config.check_row_bounds(Ident::Lhs) {
            if x_offset
                > tensor.shape(tensor.rank() - 2) - config.tiling_dimensions(Ident::Lhs).total_row()
            {
                stage.clear::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
            }
        }
        #[allow(clippy::collapsible_if)]
        if config.check_col_bounds(Ident::Rhs) {
            if y_offset
                > tensor.shape(tensor.rank() - 1) - config.tiling_dimensions(Ident::Rhs).total_col()
            {
                stage.clear::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
            }
        }

        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLoader::<MP, S, L> {
            tensor_view,
            stage,
            ident,
            _phantom: PhantomData::<(S, L)>,
        }
    }

    pub fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_full::<MP::EI, MP::ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            this.ident,
            config,
        );
    }

    pub fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(this.ident, config.to_smm_config())
    }

    pub fn reader(this: &Self) -> FullReader<MP::ES, L::TilingLayout> {
        FullReader::new(this.stage, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }
}

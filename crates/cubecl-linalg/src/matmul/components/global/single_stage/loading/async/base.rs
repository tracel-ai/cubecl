use std::marker::PhantomData;

use crate::matmul::components::global::single_stage::{self, AsyncFullLoader, FullLoader};
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
pub struct AsyncLhsLoader<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct AsyncRhsLoader<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    #[cube(comptime)]
    _config: PhantomData<S>,
    #[cube(comptime)]
    _loading: PhantomData<L>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    AsyncFullLoader<MP, single_stage::Config<S>> for AsyncLhsLoader<MP, S, L>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_full::<MP::EI, MP::ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            Ident::Lhs,
            config,
        );
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    FullLoader<MP, single_stage::Config<S>> for AsyncLhsLoader<MP, S, L>
{
    type StageReader = FullReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        FullReader::new(this.stage, Ident::Lhs)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    AsyncLhsLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let mut stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());

        #[allow(clippy::collapsible_if)]
        if config.check_row_bounds(Ident::Lhs) {
            if x_offset
                > tensor.shape(tensor.rank() - 2) - config.tiling_dimensions(Ident::Lhs).total_row()
            {
                stage.clear::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
            }
        }

        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLhsLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    FullLoader<MP, single_stage::Config<S>> for AsyncRhsLoader<MP, S, L>
{
    type StageReader = FullReader<MP::ES, L::TilingLayout>;

    fn reader(this: &Self) -> Self::StageReader {
        FullReader::new(this.stage, Ident::Rhs)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    AsyncFullLoader<MP, single_stage::Config<S>> for AsyncRhsLoader<MP, S, L>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_full::<MP::EI, MP::ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            Ident::Rhs,
            config,
        );
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: AsyncFullLoadingStrategy>
    AsyncRhsLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let mut stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());

        #[allow(clippy::collapsible_if)]
        if config.check_row_bounds(Ident::Lhs) {
            if y_offset
                > tensor.shape(tensor.rank() - 1) - config.tiling_dimensions(Ident::Rhs).total_col()
            {
                stage.clear::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
            }
        }

        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncRhsLoader::<MP, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
            _loading: PhantomData::<L>,
        }
    }
}

use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{single_stage, AsyncInputLoader, InputLoader};
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::{Barrier, BarrierLevel};
use cubecl_core::prelude::pipeline::Pipeline;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
pub trait CopyMechanism<ES: Numeric>: CubeType + Sync + Send + 'static {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>);
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Pipeline<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Barrier<ES> {
    fn memcpy_async(this: &Self, source: &Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
        this.memcpy_async(source, destination)
    }
}

#[cube]
pub trait AsyncLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout into which the loader will fill the stage
    type TilingLayout: TilingLayout;

    /// Load the full stage
    fn load_full<EG: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    /// Load the stage only at the buffer identified by buffer_index
    fn load_buffer<EG: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        buffer_index: u32,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct AsyncLhsLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
{
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct AsyncRhsLoader<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
{
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES, L::TilingLayout>,
    _config: PhantomData<S>,
    _loading: PhantomData<L>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    AsyncInputLoader<EG, ES, single_stage::Config<S>> for AsyncLhsLoader<EG, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_full::<EG, ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            Ident::Lhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    InputLoader<EG, ES, single_stage::Config<S>> for AsyncLhsLoader<EG, ES, S, L>
{
    type StageReader = LhsReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    AsyncLhsLoader<EG, ES, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
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

        AsyncLhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    InputLoader<EG, ES, single_stage::Config<S>> for AsyncRhsLoader<EG, ES, S, L>
{
    type StageReader = RhsReader<ES, L::TilingLayout>;

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    AsyncInputLoader<EG, ES, single_stage::Config<S>> for AsyncRhsLoader<EG, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load_full::<EG, ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage,
            mechanism,
            Ident::Rhs,
            config,
        );
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    AsyncRhsLoader<EG, ES, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<EG>,
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

        AsyncRhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

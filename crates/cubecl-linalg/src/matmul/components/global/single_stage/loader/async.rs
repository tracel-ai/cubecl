use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::{single_stage, AsyncInputLoader, InputLoader};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage, TilingLayout};
use crate::matmul::components::{global, Ident};
use crate::tensor::VirtualTensor;
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::pipeline::Pipeline;
use cubecl_core::prelude::*;

#[cube]
pub trait CopyMechanism<ES: Numeric>: CubeType {
    fn memcpy_async(
        mechanism: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    );
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Pipeline<ES> {
    fn memcpy_async(
        mechanism: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    ) {
        mechanism.memcpy_async(source, destination)
    }
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for Barrier<ES> {
    fn memcpy_async(
        mechanism: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    ) {
        mechanism.memcpy_async(source, destination)
    }
}

#[derive(CubeType, Copy, Clone)]
pub struct DummyLoader<ES: Numeric> {
    _e: PhantomData<ES>,
}

impl<ES: Numeric> Default for DummyLoader<ES> {
    fn default() -> Self {
        Self::new()
    }
}

#[cube]
impl<ES: Numeric> DummyLoader<ES> {
    pub fn new() -> Self {
        DummyLoader::<ES> {
            _e: PhantomData::<ES>.runtime(),
        }
    }
}

#[cube]
impl<ES: Numeric> CopyMechanism<ES> for DummyLoader<ES> {
    fn memcpy_async(
        _mechanism: &Self,
        source: &Slice<Line<ES>>,
        destination: &mut SliceMut<Line<ES>>,
    ) {
        for i in 0..source.len() {
            destination[i] = source[i];
        }
    }
}

#[cube]
pub trait AsyncLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    type TilingLayout: TilingLayout;

    fn load<EG: Numeric, ES: Numeric, G: global::GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        mechanism: CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    );
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
        mechanism: CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load::<EG, ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
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
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
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
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::StageConfig, L: AsyncLoadingStrategy>
    AsyncInputLoader<EG, ES, single_stage::Config<S>> for AsyncRhsLoader<EG, ES, S, L>
{
    fn fill_stage<CM: CopyMechanism<ES>>(
        this: &mut Self,
        mechanism: CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        L::load::<EG, ES, single_stage::Config<S>, CM>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
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
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncRhsLoader::<EG, ES, S, L> {
            tensor_view,
            stage,
            _config: PhantomData::<S>.runtime(),
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{full_load, Config, Loader, LoadingStrategy};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{global, Ident};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
    _loading: PhantomData<L>,
}

// What we want
// - Loader trait should come back
// - FullLoad's associated Loader is what we see here
// - Buffered's associated Loader is actually two loaders
// - The index of the loader is in its new, which is not in the trait

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES, LoadBuffer = Array<Line<EG>>>>
    Loader<EG, ES> for LhsLoader<EG, ES, L>
{
    type StageReader = LhsReader<ES>;
    type LoadBuffer = Array<Line<EG>>;

    fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> Self::LoadBuffer {
        L::fetch::<G>(&this.tensor_view, Ident::Lhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: Self::LoadBuffer,
        #[comptime] config: G,
    ) -> Self::StageReader {
        L::store::<G>(buffer, &mut this.stage.as_slice_mut(), Ident::Lhs, config);
        LhsReader::<ES> { stage: this.stage }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Lhs;
        let k_offset = config.stage_dim(ident).num_elements_y_dim();
        this.tensor_view.update_view(k_offset, ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> LhsLoader<EG, ES, L> {
    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: full_load::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        LhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> Loader<EG, ES> for RhsLoader<EG, ES, L> {
    type StageReader = RhsReader<ES>;
    type LoadBuffer = L::LoadBuffer;

    fn fetch_global<G: global::Config>(this: &Self, #[comptime] config: G) -> Self::LoadBuffer {
        L::fetch::<G>(&this.tensor_view, Ident::Rhs, config)
    }

    fn fill_stage<G: global::Config>(
        this: &mut Self,
        buffer: Self::LoadBuffer,
        #[comptime] config: G,
    ) -> Self::StageReader {
        L::store::<G>(buffer, &mut this.stage.as_slice_mut(), Ident::Rhs, config);
        RhsReader::<ES> { stage: this.stage }
    }

    fn to_next_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) {
        let ident = Ident::Rhs;
        let k_offset = config.stage_dim(ident).num_elements_x_dim();
        this.tensor_view.update_view(k_offset, ident);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, L: LoadingStrategy<EG, ES>> RhsLoader<EG, ES, L> {
    pub fn new<S: stage::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: full_load::Config<S>,
    ) -> Self {
        let stage = Stage::new::<S>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        RhsLoader::<EG, ES, L> {
            tensor_view,
            stage,
            _loading: PhantomData::<L>.runtime(),
        }
    }
}

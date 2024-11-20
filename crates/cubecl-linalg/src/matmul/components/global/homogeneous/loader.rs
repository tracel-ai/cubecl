use crate::matmul::components::global::Loader;
use crate::matmul::components::global::{
    homogeneous::cyclic_loading::CyclicLoading, AccumulatorLoader,
};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::Stage;
use crate::matmul::components::{global, Ident};
use crate::matmul::components::{global::tensor_view::TensorReader, tile};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct LhsLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
}

#[derive(CubeType)]
pub struct RhsLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: TensorReader<EG>,
    pub stage: Stage<ES>,
}

#[derive(CubeType)]
pub struct ZeroAccumulator;

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for LhsLoader<EG, ES> {
    type StageReader = LhsReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> LhsLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, nth_batch);

        LhsLoader::<EG, ES> { tensor_view, stage }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> Loader<EG, ES> for RhsLoader<EG, ES> {
    type StageReader = RhsReader<ES>;

    fn fill_stage<G: global::Config>(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
impl<O: Numeric, Acc: Numeric> AccumulatorLoader<O, Acc> for ZeroAccumulator {
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader = ();

    /// Fills the stage at the current k offset and returns a reader for it.
    fn fill_stage<G: global::Config>(
        _this: &mut Self,
        #[comptime] _config: G,
    ) -> Self::StageReader {
    }

    /// Load accumulator
    fn load<I: Numeric, Tile: tile::Matmul<I, Acc>>(
        _this: &mut Self,
        acc: &mut Tile::Accumulator,
        _n_offset: u32,
        #[comptime] config: Tile::Config,
    ) {
        Tile::zero_accumulator(acc, config);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> RhsLoader<EG, ES> {
    pub fn new<G: global::Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, nth_batch);

        RhsLoader::<EG, ES> { tensor_view, stage }
    }
}

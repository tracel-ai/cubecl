use std::marker::PhantomData;

use cubecl_core::prelude::barrier::{ArrivalToken, Barrier};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use crate::matmul::components::{
    global::{
        self, loader::r#async::CopyMechanism, single_stage, tensor_view::MappedTensorReader,
        AsyncInputLoader, GlobalConfig, InputLoader,
    },
    stage::{
        self,
        multi_buffer::{LhsReader, RhsReader},
        ContiguousTilingLayout, RowMajorTilingOrder, Stage,
    },
    Ident,
};

#[derive(CubeType)]
pub struct TmaLhsLoader<EG: Numeric, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<EG>,
    pub barrier: Barrier<EG>,
    pub stage: Stage<EG, ContiguousTilingLayout<RowMajorTilingOrder>>,
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct TmaRhsLoader<EG: Numeric, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<EG>,
    pub barrier: Barrier<EG>,
    pub stage: Stage<EG, ContiguousTilingLayout<RowMajorTilingOrder>>,
    _config: PhantomData<S>,
}

#[cube]
impl<EG: Numeric, S: stage::StageConfig> AsyncInputLoader<EG, EG, single_stage::Config<S>>
    for TmaLhsLoader<EG, S>
{
    fn fill_stage<CM: CopyMechanism<EG>>(
        this: &mut Self,
        _mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        let mut token = ArrivalToken::new();
        if UNIT_POS == 0 {
            let mut stage = this.stage.as_slice_mut();
            this.barrier.memcpy_async_bulk_to_shared_3d(
                &this.tensor_view.tensor,
                &mut stage,
                this.tensor_view.batch as i32,
                this.tensor_view.tile_y as i32,
                this.tensor_view.tile_x as i32,
            );
            this.barrier.arrive_tx(
                1,
                config.tiling_dimensions(Ident::Lhs).total_size() * EG::elem_size(),
                &mut token,
            );
        } else {
            this.barrier.arrive(&mut token);
        }
        this.barrier.wait(token);
    }
}

#[cube]
impl<EG: Numeric, S: stage::StageConfig> InputLoader<EG, EG, single_stage::Config<S>>
    for TmaLhsLoader<EG, S>
{
    type StageReader = LhsReader<EG, ContiguousTilingLayout<RowMajorTilingOrder>>;

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
impl<EG: Numeric, S: stage::StageConfig> TmaLhsLoader<EG, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<EG, 3>,
        x: u32,
        y: u32,
        batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);
        let barrier = Barrier::new_proxied(BarrierLevel::cube_coop(0u32));

        TmaLhsLoader::<EG, S> {
            tensor_view,
            barrier,
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<EG: Numeric, S: stage::StageConfig> InputLoader<EG, EG, single_stage::Config<S>>
    for TmaRhsLoader<EG, S>
{
    type StageReader = RhsReader<EG, ContiguousTilingLayout<RowMajorTilingOrder>>;

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
impl<EG: Numeric, S: stage::StageConfig> AsyncInputLoader<EG, EG, single_stage::Config<S>>
    for TmaRhsLoader<EG, S>
{
    fn fill_stage<CM: CopyMechanism<EG>>(
        this: &mut Self,
        _mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        let mut token = ArrivalToken::new();
        if UNIT_POS == 0 {
            let mut stage = this.stage.as_slice_mut();
            this.barrier.memcpy_async_bulk_to_shared_3d(
                &this.tensor_view.tensor,
                &mut stage,
                this.tensor_view.batch as i32,
                this.tensor_view.tile_y as i32,
                this.tensor_view.tile_x as i32,
            );
            this.barrier.arrive_tx(
                1,
                config.tiling_dimensions(Ident::Rhs).total_size() * EG::elem_size(),
                &mut token,
            );
        } else {
            this.barrier.arrive(&mut token);
        }
        this.barrier.wait(token);
    }
}

#[cube]
impl<EG: Numeric, S: stage::StageConfig> TmaRhsLoader<EG, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<EG, 3>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x_offset, y_offset, batch_offset);
        let barrier = Barrier::new_proxied(BarrierLevel::cube_coop(0u32));

        TmaRhsLoader::<EG, S> {
            tensor_view,
            barrier,
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

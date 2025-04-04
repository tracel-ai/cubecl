use core::marker::PhantomData;

use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::CopyMechanism;
use crate::matmul::components::{
    Ident,
    global::{
        self, GlobalConfig,
        single_stage::{self, AsyncFullLoader, FullLoader},
        tensor_view::MappedTensorReader,
    },
    stage::{self, ContiguousTilingLayout, RowMajorTilingOrder, Stage, multi_buffer::FullReader},
};

#[derive(CubeType)]
pub struct TmaLhsLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub barrier: Barrier<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct TmaRhsLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub barrier: Barrier<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> AsyncFullLoader<MP, single_stage::Config<S>>
    for TmaLhsLoader<MP, S>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        _mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            let mut stage = this.stage.as_slice_mut().try_cast_unchecked();
            this.barrier.memcpy_async_tensor_to_shared_3d(
                &this.tensor_view.tensor,
                &mut stage,
                this.tensor_view.batch as i32,
                this.tensor_view.tile_y as i32,
                this.tensor_view.tile_x as i32,
            );
            this.barrier.arrive_tx(
                1,
                config.tiling_dimensions(Ident::Lhs).total_size() * MP::EI::elem_size(),
            );
        } else {
            this.barrier.arrive();
        }
        this.barrier.wait();
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> FullLoader<MP, single_stage::Config<S>>
    for TmaLhsLoader<MP, S>
{
    type StageReader = FullReader<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>;

    fn reader(this: &Self) -> Self::StageReader {
        FullReader::new(this.stage, Ident::Lhs)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Lhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> TmaLhsLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x: u32,
        y: u32,
        batch: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new_aligned::<G::SmmConfig>(Ident::Lhs, 128u32, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);
        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        TmaLhsLoader::<MP, S> {
            tensor_view,
            barrier,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> FullLoader<MP, single_stage::Config<S>>
    for TmaRhsLoader<MP, S>
{
    type StageReader = FullReader<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>;

    fn reader(this: &Self) -> Self::StageReader {
        FullReader::new(this.stage, Ident::Rhs)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> AsyncFullLoader<MP, single_stage::Config<S>>
    for TmaRhsLoader<MP, S>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        _mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            let mut stage = this.stage.as_slice_mut().try_cast_unchecked();
            this.barrier.memcpy_async_tensor_to_shared_3d(
                &this.tensor_view.tensor,
                &mut stage,
                this.tensor_view.batch as i32,
                this.tensor_view.tile_y as i32,
                this.tensor_view.tile_x as i32,
            );
            this.barrier.arrive_tx(
                1,
                config.tiling_dimensions(Ident::Rhs).total_size() * MP::EI::elem_size(),
            );
        } else {
            this.barrier.arrive();
        }
        this.barrier.wait();
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Rhs, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> TmaRhsLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new_aligned::<G::SmmConfig>(Ident::Rhs, 128u32, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x_offset, y_offset, batch_offset);
        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        TmaRhsLoader::<MP, S> {
            tensor_view,
            barrier,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

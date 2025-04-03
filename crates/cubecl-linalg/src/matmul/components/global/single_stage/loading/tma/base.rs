use core::marker::PhantomData;

use cubecl_core::prelude::barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::CopyMechanism;
use crate::matmul::components::stage::StageReader;
use crate::matmul::components::{
    Ident,
    global::{
        self, GlobalConfig,
        single_stage::{self, Loader},
        tensor_view::MappedTensorReader,
    },
    stage::{self, ContiguousTilingLayout, RowMajorTilingOrder, Stage},
};

#[derive(CubeType)]
pub struct TmaLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub barrier: Barrier<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    ident: Ident,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> TmaLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x: u32,
        y: u32,
        batch: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new_aligned::<G::SmmConfig>(ident, 128u32, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);
        let barrier = Barrier::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        TmaLoader::<MP, S> {
            tensor_view,
            barrier,
            stage,
            ident,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage<CM: CopyMechanism<MP::ES>>(
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
                comptime!(config.tiling_dimensions(this.ident).total_size() * MP::EI::elem_size()),
            );
        } else {
            this.barrier.arrive();
        }
        this.barrier.wait();
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> Loader<MP, single_stage::Config<S>>
    for TmaLoader<MP, S>
{
    type TilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

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

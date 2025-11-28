use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_matmul::components::{
    MatrixPrecision,
    global::memory::{GlobalIterator, ViewDirection},
    stage::StageMemoryConfig,
};
use cubecl_std::tensor::{View, layout::Coords2d};

use cubecl_matmul::components::stage::RowMajorTilingOrder;
use cubecl_matmul::components::stage::{ContiguousTilingLayout, StridedStageMemory};

pub type TmaWeightTiling = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type TmaWeightStage<IP> = StridedStageMemory<<IP as MatrixPrecision>::Stage, TmaWeightTiling>;

#[derive(CubeType)]
pub struct TmaWeightGlobalReader<IP: MatrixPrecision> {
    pub global_iter: GlobalIterator<Line<IP::Global>>,
    pub stages: Sequence<StridedStageMemory<IP::Stage, TmaWeightTiling>>,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> TmaWeightGlobalReader<IP> {
    pub fn new(
        global_view: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] num_stages: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        let mut stages = Sequence::new();

        #[unroll]
        for _ in 0..num_stages {
            stages.push(StridedStageMemory::new_aligned(128u32, config));
        }

        let global_iter = GlobalIterator::new(global_view, k_step, ViewDirection::Row, false);

        TmaWeightGlobalReader::<IP> {
            global_iter,
            stages,
            config,
        }
    }

    pub fn fill_stage(&mut self, barrier: &Barrier, #[comptime] stage_idx: u32) {
        let stage = self.stages.index_mut(stage_idx);
        let config = comptime![self.config];

        if UNIT_POS == 0 {
            let global_view = self.global_iter.view();

            let mut stage = stage.as_slice_mut(1u32);
            let slice_size =
                config.elements_per_stage_along_col() * config.elements_per_tile_along_row;

            #[unroll]
            for tile_k in 0..config.tiles_per_stage_along_row() {
                let slice_start = slice_size * tile_k;
                let slice = stage.slice_mut(slice_start, slice_size);

                let k = tile_k * config.elements_per_tile_along_row;
                global_view.tensor_map_load(barrier, &mut slice.try_cast_unchecked(), (k, 0));
            }
        }
    }

    pub fn stage(&self, #[comptime] stage_idx: u32) -> TmaWeightStage<IP> {
        *self.stages.index(stage_idx)
    }

    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }
}

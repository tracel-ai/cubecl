use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_stage::TilingOrder;
use crate::matmul::stage_info::{tile_num_elements, total_num_elements, StageInfo};
use crate::matmul::subroutine::{PlaneMapper, SubRoutine};
use crate::matmul::tests::matmul_test_launcher::LINE_SIZE_IN;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait SharedMemoryLoader: Clone + Copy + Send + Sync + 'static {
    fn load_shared_memory<EG: Numeric, ES: Numeric, G: GlobalView<EG>, O: TilingOrder>(
        gmem: &G,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] block_info: StageInfo,
    );
}

#[derive(CubeType, Clone, Copy)]
pub struct Gmem2SmemContinuous {}

#[cube]
impl PlaneMapper for Gmem2SmemContinuous {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        CUBE_DIM_Y
    }

    fn plane_dim() -> u32 {
        CUBE_DIM_X
    }
}

#[derive(CubeType)]
pub struct Gmem2SmemProblem {
    num_stage_elements: u32,
    num_planes: u32,
    line_size: u32,
    plane_dim: u32,
}

#[cube]
impl SubRoutine for Gmem2SmemContinuous {
    type ProblemDefinition = Gmem2SmemProblem;

    fn assert_can_process(problem: Gmem2SmemProblem) {
        let jump_length = problem.num_planes * problem.line_size * problem.plane_dim;
        let can_process = problem.num_stage_elements % jump_length == 0;

        if !can_process {
            // PANIC
            return;
        }
    }
}

#[cube]
impl SharedMemoryLoader for Gmem2SmemContinuous {
    fn load_shared_memory<EG: Numeric, ES: Numeric, G: GlobalView<EG>, O: TilingOrder>(
        gmem: &G,
        smem: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    ) {
        let line_size = LINE_SIZE_IN;
        let num_stage_elements = comptime!(total_num_elements(stage_info));

        // Could be comptime if we were able to fetch line_size as comptime
        let jump_length = Self::num_planes() * line_size * Self::plane_dim();

        Gmem2SmemContinuous::assert_can_process(Gmem2SmemProblem {
            num_stage_elements,
            num_planes: Self::num_planes(),
            line_size,
            plane_dim: Self::plane_dim(),
        });

        let unit_position_base =
            (Self::plane_id() * Self::plane_dim() + Self::plane_unit()) * line_size;

        for i in 0..num_stage_elements / jump_length {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = tile_num_elements(stage_info);
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) =
                O::to_x_y(nth_tile, stage_info.num_tiles_x, stage_info.num_tiles_y);

            let line = G::load_coalesced(
                gmem,
                tile_x,
                tile_y,
                pos_within_tile,
                stage_info.tile_size_x,
                stage_info.tile_size_y,
            );

            smem[unit_position / line_size] = Line::cast_from(line);
        }
    }
}

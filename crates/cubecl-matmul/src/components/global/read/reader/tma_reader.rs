use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, StridedStage, TilingOrder,
};
use crate::components::stage::{RowMajorTilingOrder, StageMemoryConfig, TilingOrderEnum};
use crate::components::{MatmulIdent, MatrixPrecision};
use crate::components::{MatrixLayout, global::memory::GlobalIterator};

/// TMA uses contiguous tiling, but with a special tiling order
pub type TmaTilingLayout = ContiguousTilingLayout<TmaTilingOrder>;
/// TMA uses standard full stage to tile reader
pub type TmaStage<IP> = StridedStage<<IP as MatrixPrecision>::Stage, TmaTilingLayout>;

#[derive(CubeType, Clone, Copy)]
/// A special tiling order where:
/// - If the matrix data layout is row-major, the tiling order is col-major
/// - If the matrix data layout is col-major, the tiling order is row-major
pub struct TmaTilingOrder;

#[cube]
impl TilingOrder for TmaTilingOrder {
    fn to_row_col(
        nth: u32,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> Coords2d {
        match config.matrix_layout {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_row_col(nth, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_row_col(nth, tile_count_rows, tile_count_cols, config)
            }
        }
    }

    fn to_nth_tile(
        tile: Coords2d,
        tile_count_rows: u32,
        tile_count_cols: u32,
        #[comptime] config: StageMemoryConfig,
    ) -> u32 {
        match config.matrix_layout {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_nth_tile(tile, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_nth_tile(tile, tile_count_rows, tile_count_cols, config)
            }
        }
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::Tma
    }
}

#[derive(CubeType)]
/// Loads the entire stage memory using TMA (Tensor Memory Accelerator)
pub struct TmaGlobalReader<IP: MatrixPrecision> {
    global_iter: GlobalIterator<Line<IP::Global>>,
    stage: StridedStage<IP::Stage, TmaTilingLayout>,
    #[cube(comptime)]
    config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> TmaGlobalReader<IP> {
    /// Create a TmaGlobalReader
    pub fn new(
        global_view: View<Line<IP::Global>, Coords2d>,
        k_step: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        let global_iter = GlobalIterator::new(global_view, k_step, ident.view_direction(), false);
        let stage = StridedStage::new_aligned(128u32, config);

        TmaGlobalReader::<IP> {
            global_iter,
            stage,
            config,
        }
    }

    /// Load data into the full stage memory
    /// We load contiguous "pillars" to ensure contiguous tiling while minimizing the
    /// number of load calls. Naming assumes row-major, for col-major the dimensions are transposed.
    pub fn load_stage(&mut self, barrier: &Barrier) {
        if UNIT_POS == 0 {
            let config = comptime![self.config];

            let size_row = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_in_stage_row(),
                MatrixLayout::ColMajor => config.elements_in_stage_col(),
            };
            let size_col = match config.matrix_layout {
                MatrixLayout::RowMajor => config.elements_in_tile_col,
                MatrixLayout::ColMajor => config.elements_in_tile_row,
            };
            let tile_count_col = match config.matrix_layout {
                MatrixLayout::RowMajor => config.tiles_in_stage_col,
                MatrixLayout::ColMajor => config.tiles_in_stage_row,
            };

            let global_view = self.global_iter.view();
            let mut stage = self.stage.as_slice_mut(1u32);
            let slice_size = size_row * size_col;

            #[unroll]
            for tile_col in 0..tile_count_col {
                let slice_start = tile_col * slice_size;
                let slice = stage.slice_mut(slice_start, slice_start + slice_size);
                let col = tile_col * size_col;

                let pos = match config.matrix_layout {
                    MatrixLayout::RowMajor => (0, col),
                    MatrixLayout::ColMajor => (col, 0),
                };

                global_view.tensor_map_load(barrier, &mut slice.try_cast_unchecked(), pos);
            }
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn stage(&self) -> TmaStage<IP> {
        self.stage
    }

    /// Advance the view over global memory along the k dimension.
    pub fn advance_view(&mut self) {
        self.global_iter.advance();
    }
}

#[cube]
/// Barrier for TMA
pub fn arrive_tma(barrier: &Barrier, #[comptime] num_bytes: u32) {
    if UNIT_POS == 0 {
        barrier.arrive_tx(1, num_bytes);
    } else {
        barrier.arrive();
    }
}

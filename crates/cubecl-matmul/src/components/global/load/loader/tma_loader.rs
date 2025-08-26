use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use crate::components::MatrixLayout;
use crate::components::stage::{
    FullStageToTileReader, RowMajorTilingOrder, StageMemoryConfig, TilingOrderEnum,
};
use crate::components::{InputPrecision, MatmulIdent, StageIdent};
use crate::components::{
    global::{GlobalConfig, memory::MappedTensorReader},
    stage::{ColMajorTilingOrder, ContiguousTilingLayout, StageMemory, TilingOrder},
};

/// TMA uses contiguous tiling, but with a special tiling order
pub type TmaTiling = ContiguousTilingLayout<TmaTilingOrder>;
/// TMA uses standard full stage to tile reader
pub type TmaReader<IP> = FullStageToTileReader<<IP as InputPrecision>::Stage, TmaTiling>;

#[derive(CubeType, Clone, Copy)]
/// A special tiling order where:
/// - If the matrix data layout is row-major, the tiling order is col-major
/// - If the matrix data layout is col-major, the tiling order is row-major
pub struct TmaTilingOrder;

#[cube]
impl TilingOrder for TmaTilingOrder {
    fn to_row_col<C: StageMemoryConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: C,
    ) -> (u32, u32) {
        match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => ColMajorTilingOrder::to_row_col::<C>(
                nth,
                tile_count_rows,
                tile_count_cols,
                ident,
                config,
            ),
            MatrixLayout::ColMajor => RowMajorTilingOrder::to_row_col::<C>(
                nth,
                tile_count_rows,
                tile_count_cols,
                ident,
                config,
            ),
        }
    }

    fn to_nth_tile<C: StageMemoryConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: C,
    ) -> u32 {
        match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => ColMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                ident,
                config,
            ),
            MatrixLayout::ColMajor => RowMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                ident,
                config,
            ),
        }
    }

    fn to_enum() -> comptime_type!(TilingOrderEnum) {
        TilingOrderEnum::Tma
    }
}

#[derive(CubeType)]
/// Loads the entire stage memory using TMA (Tensor Memory Accelerator)
pub struct TmaLoader<IP: InputPrecision, G: GlobalConfig> {
    pub tensor_view: MappedTensorReader<IP::Global>,
    pub stage: StageMemory<IP::Stage, TmaTiling>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<IP: InputPrecision, G: GlobalConfig> TmaLoader<IP, G> {
    /// Create a TmaLoader
    pub fn new(
        tensor: TensorMap<IP::Global>,
        x: u32,
        y: u32,
        batch: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage = StageMemory::new_aligned::<G::StageMemoryConfig>(
            comptime!(ident.into_stage()),
            128u32,
            config.stage_memory_config(),
        );

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);

        TmaLoader::<IP, G> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData,
        }
    }

    /// Fill the full stage memory
    pub fn fill_stage(this: &mut Self, barrier: &Barrier, #[comptime] config: G) {
        if UNIT_POS == 0 {
            let ident = comptime!(this.ident);
            let stage_ident = comptime!(ident.into_stage());
            // The tensor map is encoded as the transposed shape, so we need to swap coordinates
            let (row, col) = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => (this.tensor_view.tile_x, this.tensor_view.tile_y),
                MatrixLayout::ColMajor => (this.tensor_view.tile_y, this.tensor_view.tile_x),
            };

            let size_row = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tiling_scheme().elements_in_stage_row(stage_ident),
                MatrixLayout::ColMajor => config.tiling_scheme().elements_in_stage_col(stage_ident),
            };
            let size_col = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tiling_scheme().elements_in_tile_col(stage_ident),
                MatrixLayout::ColMajor => config.tiling_scheme().elements_in_tile_row(stage_ident),
            };
            let tile_count_col = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => config.tiling_scheme().tiles_in_stage_col(stage_ident),
                MatrixLayout::ColMajor => config.tiling_scheme().tiles_in_stage_row(stage_ident),
            };

            let tensor = this.tensor_view.tensor.try_cast_unchecked();
            let mut stage = this.stage.as_slice_mut(1u32);
            let slice_size = size_row * size_col;
            let batch = this.tensor_view.batch as i32;

            #[unroll]
            for tile_col in 0..tile_count_col {
                let slice_start = tile_col * slice_size;
                let mut slice = stage.slice_mut(slice_start, slice_start + slice_size);
                let col = col + tile_col * size_col;

                barrier.tma_load_3d(&tensor, &mut slice, batch, row as i32, col as i32);
            }
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn reader(this: &Self) -> TmaReader<IP> {
        TmaReader::<IP>::new(this.stage, comptime!(this.ident.into_stage()))
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
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

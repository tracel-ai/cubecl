use core::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    Ident,
    global::{
        self, GlobalConfig,
        single_stage::{self, AsyncFullLoader, FullLoader},
        tensor_view::MappedTensorReader,
    },
    stage::{
        self, ColMajorTilingOrder, ContiguousTilingLayout, Stage, StageConfig, TilingOrder,
        multi_buffer::{LhsReader, RhsReader},
    },
};
use crate::matmul::components::{MatmulPrecision, MatrixLayout};
use crate::matmul::components::{global::CopyMechanism, stage::RowMajorTilingOrder};

pub type TmaTilingLhs = ContiguousTilingLayout<TmaTilingOrderLhs>;
pub type TmaTilingRhs = ContiguousTilingLayout<TmaTilingOrderRhs>;

#[derive(CubeType, Clone, Copy)]
pub struct TmaTilingOrderLhs {}
#[derive(CubeType, Clone, Copy)]
pub struct TmaTilingOrderRhs {}

#[cube]
impl TilingOrder for TmaTilingOrderLhs {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] config: C,
    ) -> (u32, u32) {
        match config.matrix_layout(Ident::Lhs) {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_row_col::<C>(nth, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_row_col::<C>(nth, tile_count_rows, tile_count_cols, config)
            }
        }
    }
    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] config: C,
    ) -> u32 {
        match config.matrix_layout(Ident::Lhs) {
            MatrixLayout::RowMajor => ColMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                config,
            ),
            MatrixLayout::ColMajor => RowMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                config,
            ),
        }
    }
}

#[cube]
impl TilingOrder for TmaTilingOrderRhs {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] config: C,
    ) -> (u32, u32) {
        match config.matrix_layout(Ident::Rhs) {
            MatrixLayout::RowMajor => {
                ColMajorTilingOrder::to_row_col::<C>(nth, tile_count_rows, tile_count_cols, config)
            }
            MatrixLayout::ColMajor => {
                RowMajorTilingOrder::to_row_col::<C>(nth, tile_count_rows, tile_count_cols, config)
            }
        }
    }
    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] config: C,
    ) -> u32 {
        match config.matrix_layout(Ident::Rhs) {
            MatrixLayout::RowMajor => ColMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                config,
            ),
            MatrixLayout::ColMajor => RowMajorTilingOrder::to_nth_tile::<C>(
                row,
                col,
                tile_count_rows,
                tile_count_cols,
                config,
            ),
        }
    }
}

#[derive(CubeType)]
pub struct TmaLhsLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: Stage<MP::ES, TmaTilingLhs>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[derive(CubeType)]
pub struct TmaRhsLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: Stage<MP::ES, TmaTilingRhs>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> AsyncFullLoader<MP, single_stage::Config<S>>
    for TmaLhsLoader<MP, S>
{
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        barrier: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            // The tensor map is encoded as the transposed shape, so we need to swap coordinates
            let (row, col) = match config.matrix_layout(Ident::Lhs) {
                MatrixLayout::RowMajor => (this.tensor_view.tile_x, this.tensor_view.tile_y),
                MatrixLayout::ColMajor => (this.tensor_view.tile_y, this.tensor_view.tile_x),
            };

            let tiling_dims = config.tiling_dimensions(Ident::Lhs);
            let size_row = match config.matrix_layout(Ident::Lhs) {
                MatrixLayout::RowMajor => tiling_dims.total_row(),
                MatrixLayout::ColMajor => tiling_dims.total_col(),
            };
            let size_col = match config.matrix_layout(Ident::Lhs) {
                MatrixLayout::RowMajor => tiling_dims.tile_shape_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_shape_row(),
            };
            let tile_count_col = match config.matrix_layout(Ident::Lhs) {
                MatrixLayout::RowMajor => tiling_dims.tile_count_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_count_row(),
            };

            let tensor = this.tensor_view.tensor.try_cast_unchecked();
            let mut stage = this.stage.as_slice_mut();
            let slice_size = size_row * size_col;

            #[unroll]
            for tile_col in 0..tile_count_col {
                let slice_start = tile_col * slice_size;
                let mut slice = stage.slice_mut(slice_start, slice_start + slice_size);
                let col = col + tile_col * size_col;
                CM::tma_load_3d(
                    barrier,
                    &tensor,
                    &mut slice,
                    this.tensor_view.batch,
                    row,
                    col,
                );
            }
        }
    }

    fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(Ident::Lhs, config.to_smm_config())
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> FullLoader<MP, single_stage::Config<S>>
    for TmaLhsLoader<MP, S>
{
    type StageReader = LhsReader<MP::ES, TmaTilingLhs>;

    fn reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
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

        TmaLhsLoader::<MP, S> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig> FullLoader<MP, single_stage::Config<S>>
    for TmaRhsLoader<MP, S>
{
    type StageReader = RhsReader<MP::ES, TmaTilingRhs>;

    fn reader(this: &Self) -> Self::StageReader {
        RhsReader::new(this.stage)
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
        barrier: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            // The tensor map is encoded as the transposed shape, so we need to swap coordinates
            let (row, col) = match config.matrix_layout(Ident::Rhs) {
                MatrixLayout::RowMajor => (this.tensor_view.tile_x, this.tensor_view.tile_y),
                MatrixLayout::ColMajor => (this.tensor_view.tile_y, this.tensor_view.tile_x),
            };

            let tiling_dims = config.tiling_dimensions(Ident::Rhs);
            let size_row = match config.matrix_layout(Ident::Rhs) {
                MatrixLayout::RowMajor => tiling_dims.total_row(),
                MatrixLayout::ColMajor => tiling_dims.total_col(),
            };
            let size_col = match config.matrix_layout(Ident::Rhs) {
                MatrixLayout::RowMajor => tiling_dims.tile_shape_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_shape_row(),
            };
            let tile_count_col = match config.matrix_layout(Ident::Rhs) {
                MatrixLayout::RowMajor => tiling_dims.tile_count_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_count_row(),
            };

            let tensor = this.tensor_view.tensor.try_cast_unchecked();
            let mut stage = this.stage.as_slice_mut();
            let slice_size = size_row * size_col;

            #[unroll]
            for tile_col in 0..tile_count_col {
                let slice_start = tile_col * slice_size;
                let mut slice = stage.slice_mut(slice_start, slice_start + slice_size);
                let col = col + tile_col * size_col;
                CM::tma_load_3d(
                    barrier,
                    &tensor,
                    &mut slice,
                    this.tensor_view.batch,
                    row,
                    col,
                );
            }
        }
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

        TmaRhsLoader::<MP, S> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
        }
    }
}

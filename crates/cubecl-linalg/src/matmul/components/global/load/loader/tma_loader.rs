use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::CubeOption;

use crate::matmul::components::stage::{FullReader, RowMajorTilingOrder};
use crate::matmul::components::{
    Ident, InputIdent, MatmulPrecision, MatrixLayout,
    global::{Quantization, single_stage},
};
use crate::matmul::components::{
    global::{self, GlobalConfig, tensor_view::MappedTensorReader},
    stage::{
        self, ColMajorTilingOrder, ContiguousTilingLayout, StageConfig, StageMemory, TilingOrder,
    },
};

pub type TmaTiling = ContiguousTilingLayout<TmaTilingOrder>;
pub type TmaReader<MP> = FullReader<<MP as MatmulPrecision>::ES, TmaTiling>;

#[derive(CubeType, Clone, Copy)]
pub struct TmaTilingOrder;

#[cube]
impl TilingOrder for TmaTilingOrder {
    fn to_row_col<C: StageConfig>(
        nth: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
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
    fn to_nth_tile<C: StageConfig>(
        row: u32,
        col: u32,
        #[comptime] tile_count_rows: u32,
        #[comptime] tile_count_cols: u32,
        #[comptime] ident: Ident,
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
}

#[derive(CubeType)]
pub struct TmaLoader<MP: MatmulPrecision, S: stage::StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: StageMemory<MP::ES, TmaTiling>,
    #[cube(comptime)]
    ident: InputIdent,
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
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let stage = StageMemory::new_aligned::<G::SmmConfig>(
            comptime!(ident.as_ident()),
            128u32,
            config.to_smm_config(),
        );

        let tensor_view = MappedTensorReader::new(tensor, x, y, batch);

        TmaLoader::<MP, S> {
            tensor_view,
            stage,
            ident,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage(
        this: &mut Self,
        barrier: &Barrier<MP::ES>,
        #[comptime] config: single_stage::Config<S>,
    ) {
        if UNIT_POS == 0 {
            let ident = comptime!(this.ident.as_ident());
            // The tensor map is encoded as the transposed shape, so we need to swap coordinates
            let (row, col) = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => (this.tensor_view.tile_x, this.tensor_view.tile_y),
                MatrixLayout::ColMajor => (this.tensor_view.tile_y, this.tensor_view.tile_x),
            };

            let tiling_dims = config.tiling_dimensions(ident);
            let size_row = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => tiling_dims.total_row(),
                MatrixLayout::ColMajor => tiling_dims.total_col(),
            };
            let size_col = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => tiling_dims.tile_shape_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_shape_row(),
            };
            let tile_count_col = match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => tiling_dims.tile_count_col(),
                MatrixLayout::ColMajor => tiling_dims.tile_count_row(),
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

    pub fn reader(this: &Self) -> TmaReader<MP> {
        TmaReader::<MP>::new(this.stage, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view
            .update_view(k_offset, comptime!(this.ident.as_ident()));
    }
}

#[cube]
pub(crate) fn arrive_tma<E: CubePrimitive>(barrier: &Barrier<E>, #[comptime] num_elems: u32) {
    if UNIT_POS == 0 {
        barrier.arrive_tx(1, num_elems * E::elem_size());
    } else {
        barrier.arrive();
    }
}

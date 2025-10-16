use crate::components::attention_types::*;
use crate::components::tile::MaskTile;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::global::memory::{GlobalIterator, ViewDirection};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;
use cubecl_std::CubeOption;

#[derive(CubeType)]
pub struct LogicalIterator {
    row: u32,
    col: RuntimeCell<u32>,
    step_col: u32,
}

#[cube]
impl LogicalIterator {
    fn init(q_offset: u32, step_col: u32) -> LogicalIterator {
        LogicalIterator {
            row: q_offset,
            col: RuntimeCell::new(0),
            step_col,
        }
    }

    fn read(&self) -> Coords2d {
        (self.row, self.col.read())
    }

    fn advance(&mut self) {
        self.col.store(self.col.read() + self.step_col);
    }
}

#[derive(CubeType)]
pub enum MaskReader<AP: AttentionPrecision> {
    Materialized(GlobalIterator<Line<MSK<AP>>>, LogicalIterator),
    Logical(LogicalIterator),
}

#[cube]
impl<AP: AttentionPrecision> MaskReader<AP> {
    pub fn new_logical(q_offset: u32, step: u32) -> Self {
        MaskReader::<AP>::new_Logical(LogicalIterator::init(q_offset, step))
    }

    pub fn new_materialized(q_offset: u32, mask: View<Line<MSK<AP>>, Coords2d>, step: u32) -> Self {
        let mask = mask.slice((q_offset, 0), mask.shape());
        let global_iter = GlobalIterator::new(mask, step, ViewDirection::Col, false);

        MaskReader::<AP>::new_Materialized(global_iter, LogicalIterator::init(q_offset, step))
    }

    // TODO read tile too
    pub fn read(&self) -> Coords2d {
        match self {
            MaskReader::Materialized(global_iterator, logical_iterator) => logical_iterator.read(),
            MaskReader::Logical(logical_iterator) => logical_iterator.read(),
        }
    }

    pub fn advance_view(&mut self) {
        match self {
            MaskReader::Logical(logical_iter) => logical_iter.advance(),
            MaskReader::Materialized(global_iter, logical_iter) => {
                global_iter.advance();
                logical_iter.advance()
            }
        }
    }
}

// if UNIT_POS_Y == 0 {
//     // TODO this reader is bad, it's not coalesced
//     let memory_config = config.global_memory_config(AttentionIdent::Value);
//     let mut slice = self.stage_memory.as_slice_mut(1u32);

//     let tile_rows = memory_config.elements_in_tile_row;
//     let tile_cols = memory_config.elements_in_tile_col;
//     let partition_rows = memory_config.elements_in_stage_row / tile_rows;
//     let partition_cols = memory_config.elements_in_stage_col / tile_cols;

//     let units_per_tile_row = comptime!(config.plane_dim() / tile_rows);
//     let tile_cols_per_unit = comptime!(div_ceil(tile_cols, units_per_tile_row));

//     let row_in_tile = UNIT_POS_X / units_per_tile_row;
//     let col_in_tile_start = (UNIT_POS_X % units_per_tile_row) * tile_cols_per_unit;

//     // Assumes row tiling order
//     let num_elements_per_tile = tile_rows * tile_cols;
//     let tile_row_stride = partition_cols * num_elements_per_tile;
//     let tile_col_stride = num_elements_per_tile;

//     let layout = TiledLayout::new(memory_config);
//     let view = self.global_iter.view().view(layout);

//     #[unroll]
//     for tile_row in 0..partition_rows {
//         #[unroll]
//         for tile_col in 0..partition_cols {
//             if row_in_tile < tile_rows {
//                 #[unroll]
//                 for i in 0..tile_cols_per_unit {
//                     let col = col_in_tile_start + i;

//                     if col < tile_cols {
//                         let tile_row_offset = tile_row * tile_row_stride;
//                         let tile_col_offset = tile_col * tile_col_stride;
//                         let offset = tile_row_offset + tile_col_offset;

//                         let index = row_in_tile * tile_cols + col;

//                         slice[index + offset] = Line::cast_from(
//                             view.read_checked(((tile_row, tile_col), index)),
//                         );
//                     }
//                 }
//             }
//         }
//     }
// }

// pub fn get_tile<S: StageAttentionConfig>(
//     &self,
//     tile: Coords2d,
//     #[comptime] config: S,
// ) -> CubeOption<StridedTile<MSK<AP>>> {
//     match self {
//         MaskReader::Logical(logical_iter) => CubeOption::new_None(),
//         MaskReader::Materialized(global_iter, logical_iter) => {
//             let (row_in_partition, col) = tile;
//             let attention_tile_size = config.tiling_scheme().tile_size;

//             let row =
//                 row_in_partition + UNIT_POS_Y * config.tiling_scheme().partition_size.seq_q;

//             let tile = StridedTile::<MSK<AP>>::new_strided(
//                 global_iter
//                     .view()
//                     .slice(
//                         (
//                             row * attention_tile_size.seq_q,
//                             col * attention_tile_size.seq_kv,
//                         ),
//                         (attention_tile_size.seq_q, attention_tile_size.seq_kv).runtime(),
//                     )
//                     .to_linear_slice(),
//                 config.tiling_scheme().elements_in_partition_seq_kv(),
//                 MatrixLayout::RowMajor,
//             );

//             CubeOption::new_Some(tile)
//         }
//     }
// }

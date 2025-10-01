use cubecl::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};

use crate::components::{MatrixLayout, tile::StridedTile};

/// Writer for storing the output registers.
#[derive(CubeType)]
pub struct MmaStageWriter {}

#[cube]
impl MmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &Sequence<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
    ) {
        let num_lines = def.lines_per_lane(ident);
        let line_size = def.line_size(ident);
        let lane_id = UNIT_POS_PLANE;
        let (_, stride) = tile.as_unlined();
        // Supported on all targets that support manual MMA
        let mut slice = tile.slice.with_line_size(line_size);

        let (stride_row, stride_col) = match layout {
            MatrixLayout::RowMajor => (stride, 1),
            MatrixLayout::ColMajor => (1, stride),
        };

        #[unroll]
        for i in 0..num_lines {
            let value = *fragment.index(i);
            let elem_idx = i * line_size;
            let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
            let offset = row * stride_row + col * stride_col;
            slice[offset / line_size] = Line::cast_from(value);
        }
    }
}

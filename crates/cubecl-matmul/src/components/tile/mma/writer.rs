use cubecl::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};

use crate::components::{MatrixLayout, as_cmma_layout, tile::StridedTile};

/// Writer for storing the output registers.
#[derive(CubeType)]
pub struct MmaStageWriter {}

#[cube]
impl MmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
    ) {
        let line_layout = def.line_layout(ident);
        let transposed = comptime![as_cmma_layout(layout) != line_layout];

        if transposed {
            store_manual_transposed(tile, fragment, def, ident, layout);
        } else {
            store_manual_plain(tile, fragment, def, ident, layout);
        }
    }
}

#[cube]
fn store_manual_transposed<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &mut StridedTile<V, ReadWrite>,
    fragment: &Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let (_, stride) = tile.as_unlined_mut();
    let mut tile = tile.with_line_size(1u32);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        #[unroll]
        for n in 0..line_size {
            let elem_idx = i * line_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
            let offset = row * stride_row + col * stride_col;
            let offset = tile.stage_offset(offset);

            tile.stage[offset] = Line::cast_from(fragment[i][n]);
        }
    }
}

#[cube]
fn store_manual_plain<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &mut StridedTile<V, ReadWrite>,
    fragment: &Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;
    let (_, stride) = tile.as_unlined_mut();
    // Supported on all targets that support manual MMA
    let mut tile = tile.with_line_size(line_size);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let value = fragment[i];
        let elem_idx = i * line_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
        let offset = row * stride_row + col * stride_col;
        let offset = tile.stage_offset(offset / line_size);

        tile.stage[offset] = Line::cast_from(value);
    }
}

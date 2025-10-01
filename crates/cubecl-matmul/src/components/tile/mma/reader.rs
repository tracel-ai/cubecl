use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout, as_cmma_layout,
    tile::{
        StridedTile,
        io::{Filled, Strided, TileKind},
    },
};

/// Generic CMMA reader over any tile type
#[cube]
pub(crate) trait MmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &<Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut Sequence<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
    );
}

/// Reader to load the manual MMA registers. Tile kind determines implementation.
#[derive(CubeType)]
pub struct MmaStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl MmaFragmentReader for MmaStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &StridedTile<V>,
        fragment: &mut Sequence<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
    ) {
        let line_layout = def.line_layout(ident);

        let transposed = comptime![as_cmma_layout(layout) != line_layout];

        if transposed {
            load_transposed(tile, fragment, def, ident, layout);
        } else {
            load_plain(tile, fragment, def, ident, layout);
        }
    }
}

#[cube]
fn load_transposed<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V>,
    fragment: &mut Sequence<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let (_, stride) = tile.as_unlined();
    let slice = tile.slice.with_line_size(1u32);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let line = fragment.index_mut(i);
        #[unroll]
        for n in 0..line_size {
            let elem_idx = i * line_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
            let offset = row * stride_row + col * stride_col;
            line[n] = E::cast_from(slice[offset]);
        }
    }
}

#[cube]
fn load_plain<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V>,
    fragment: &mut Sequence<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;
    let (_, stride) = tile.as_unlined();
    // Supported on all targets that support manual MMA
    let slice = tile.slice.with_line_size(line_size);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let elem_idx = i * line_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
        let offset = row * stride_row + col * stride_col;

        let value = slice[offset / line_size];
        *fragment.index_mut(i) = Line::cast_from(value);
    }
}

#[cube]
impl MmaFragmentReader for MmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        value: &V,
        fragment: &mut Sequence<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] _layout: MatrixLayout,
    ) {
        let num_lines = def.lines_per_lane(ident);
        let value = Line::<E>::cast_from(*value);

        #[unroll]
        for i in 0..num_lines {
            *fragment.index_mut(i) = value;
        }
    }
}

#[cube]
impl<Inner: TileKind> MmaFragmentReader for MmaStageReader<CubeOption<Inner>>
where
    MmaStageReader<Inner>: MmaFragmentReader<TileKind = Inner>,
{
    type TileKind = CubeOption<Inner>;

    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &CubeOption<Inner::Tile<V>>,
        fragment: &mut Sequence<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                MmaStageReader::<Inner>::load_fragment(tile, fragment, def, ident, layout)
            }
            CubeOption::None => MmaStageReader::<Filled>::load_fragment::<E, V, A, B, CD>(
                &V::from_int(0),
                fragment,
                def,
                ident,
                layout,
            ),
        }
    }
}

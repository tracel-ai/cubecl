use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout, as_cmma_layout, from_cmma_layout,
    tile::{
        StridedTile,
        io::{Filled, Strided, TileKind},
        mma::config::{LoadMethod, MmaMatmulConfig},
    },
};

/// Generic CMMA reader over any tile type
#[cube]
pub(crate) trait MmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        tile: &<Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] config: MmaMatmulConfig,
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
        fragment: &mut Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] config: MmaMatmulConfig,
    ) {
        let line_layout = def.line_layout(ident);

        let transposed = comptime![as_cmma_layout(layout) != line_layout];

        match config.load_method(ident) {
            LoadMethod::Manual => {
                if transposed {
                    load_manual_transposed(tile, fragment, def, ident, layout);
                } else {
                    load_manual_plain(tile, fragment, def, ident, layout);
                }
            }
            LoadMethod::LoadMatrix => {
                load_ldmatrix(tile, fragment, def, transposed, ident, layout, config);
            }
        }
    }
}

#[cube]
fn load_manual_transposed<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V>,
    fragment: &mut Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let (_, stride) = tile.as_unlined();
    let tile = tile.with_line_size(1u32);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let mut line = Line::empty(line_size);
        #[unroll]
        for n in 0..line_size {
            let elem_idx = i * line_size + n;
            let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
            let offset = row * stride_row + col * stride_col;
            let offset = tile.stage_offset(offset);

            line[n] = E::cast_from(tile.stage[offset]);
        }
        fragment[i] = line;
    }
}

#[cube]
fn load_manual_plain<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V>,
    fragment: &mut Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_lines = def.lines_per_lane(ident);
    let line_size = def.line_size(ident);

    let lane_id = UNIT_POS_PLANE;
    let (_, stride) = tile.as_unlined();
    // Supported on all targets that support manual MMA
    let tile = tile.with_line_size(line_size);

    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    #[unroll]
    for i in 0..num_lines {
        let elem_idx = i * line_size;
        let (row, col) = def.position_of_nth(lane_id, elem_idx, ident);
        let offset = row * stride_row + col * stride_col;
        let stage_offset = tile.stage_offset(offset / line_size);

        fragment[i] = Line::cast_from(tile.stage[stage_offset]);
    }
}

/// This is important to use on CUDA because CUDA's matrices are heavily permuted, being organized
/// into 8x8 chunks with only 32 contiguous bits per thread. `ldmatrix` loads 8 consecutive elements
/// in each thread (if executed with x4), then uses warp shuffles to move the elements to the
/// correct positions for each thread. This currently only supports f16, fp8 needs more handling and
/// packed fp4 isn't supported at all. So these currently fall back to manual loading.
/// tf32 isn't supported by the instruction at all.
#[cube]
fn load_ldmatrix<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    tile: &StridedTile<V>,
    fragment: &mut Array<Line<E>>,
    def: MmaDefinition<A, B, CD>,
    #[comptime] transposed: bool,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmulConfig,
) {
    let stage_line_size = tile.stage.line_size();
    let (_, stride) = tile.as_unlined();

    let elem_size = E::type_size();
    let num_regs = def.lines_per_lane(ident);
    let width = comptime![16 / elem_size / stage_line_size];

    let start = ldmatrix_offset::<V, A, B, CD>(stride, def, stage_line_size, ident, layout, config);
    let start = tile.stage_offset(start);

    let row_slice = tile.stage.slice(start, start + width);
    let regs = def.load_matrix(&row_slice, ident, num_regs, transposed);

    #[unroll]
    for i in 0..num_regs {
        fragment[i] = Line::cast_from(regs[i]);
    }
}

/// This logic is horrible and hard to reason about, and very hardcoded. But can't figure out a
/// better way to do it.
#[cube]
pub(crate) fn ldmatrix_offset<E: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    stride: u32,
    def: MmaDefinition<A, B, CD>,
    #[comptime] stage_line_size: u32,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] config: MmaMatmulConfig,
) -> u32 {
    let expected_layout = from_cmma_layout(def.line_layout(ident));
    let tiling = config.shared.tile_size;
    let (stride_row, stride_col) = match layout {
        MatrixLayout::RowMajor => (stride, 1),
        MatrixLayout::ColMajor => (1, stride),
    };

    let elem_size = E::type_size();
    let num_regs = def.lines_per_lane(ident);
    let width = comptime![16 / elem_size];
    // Height is always 8, and lanes are divided into blocks of 8.
    let height = 8;

    let (total_rows, total_cols) = match ident {
        MatrixIdent::A => (tiling.m(), tiling.k()),
        MatrixIdent::B => (tiling.k(), tiling.n()),
        MatrixIdent::Accumulator => (tiling.m(), tiling.n()),
    };
    // tile is treated as row-major, if col-major the tile shape is just inverted
    let total_cols = match comptime![expected_layout] {
        MatrixLayout::RowMajor => total_cols,
        MatrixLayout::ColMajor => total_rows,
    };

    //  Indices are wrapped for < 4 registers.
    let lane = UNIT_POS_PLANE;
    let sub_lane = lane % height;
    let nth_matrix = lane / height % num_regs;

    let tiles_col = total_cols / height;

    // Tiles are arranged in column-major fashion
    let row_offs = (nth_matrix % tiles_col) * 8;
    let col_offs = (nth_matrix / tiles_col) * width;

    let (row, col) = match layout {
        MatrixLayout::RowMajor => (row_offs + sub_lane, col_offs),
        MatrixLayout::ColMajor => (row_offs, col_offs + sub_lane),
    };

    let start = row * stride_row + col * stride_col;
    start / stage_line_size
}

#[cube]
impl MmaFragmentReader for MmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
        value: &V,
        fragment: &mut Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] _layout: MatrixLayout,
        #[comptime] _config: MmaMatmulConfig,
    ) {
        let num_lines = def.lines_per_lane(ident);
        let value = Line::<E>::cast_from(*value);

        #[unroll]
        for i in 0..num_lines {
            fragment[i] = value;
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
        fragment: &mut Array<Line<E>>,
        def: MmaDefinition<A, B, CD>,
        #[comptime] ident: MatrixIdent,
        #[comptime] layout: MatrixLayout,
        #[comptime] config: MmaMatmulConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                MmaStageReader::<Inner>::load_fragment(tile, fragment, def, ident, layout, config)
            }
            CubeOption::None => MmaStageReader::<Filled>::load_fragment::<E, V, A, B, CD>(
                &V::from_int(0),
                fragment,
                def,
                ident,
                layout,
                config,
            ),
        }
    }
}

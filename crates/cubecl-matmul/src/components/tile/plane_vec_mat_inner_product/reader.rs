use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout,
    tile::{
        StridedTile,
        io::{Filled, Strided, Tile, TileKind},
        plane_vec_mat_inner_product::LineContainer,
    },
};

/// Reader for the vector side of the VecMat operation
#[derive(CubeType)]
pub struct VectorStageReader {}

/// Generic matrix reader over any tile type
#[cube]
pub(super) trait MatrixFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &Tile<Self::TileKind, V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] n: u32,
    );
}

/// Reader for the matrix side of the VecMat operation. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct MatrixStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl VectorStageReader {
    pub fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        frag: &mut LineContainer<E>,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        let offset = tile.stage_offset(UNIT_POS_X);
        frag.line = Line::cast_from(tile.stage[offset]);
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] n: u32,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        #[unroll]
        for n in 0..n {
            let line_container = frag.index_mut(n);
            let offset = tile.stage_offset(UNIT_POS_X + n * tile.stride);
            line_container.line = Line::cast_from(tile.stage[offset]);
        }
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric>(
        value: &V,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] n: u32,
    ) {
        #[unroll]
        for n in 0..n {
            let line_container = frag.index_mut(n);
            line_container.line = Line::cast_from(*value);
        }
    }
}

#[cube]
impl<Inner: TileKind> MatrixFragmentReader for MatrixStageReader<CubeOption<Inner>>
where
    MatrixStageReader<Inner>: MatrixFragmentReader<TileKind = Inner>,
{
    type TileKind = CubeOption<Inner>;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &CubeOption<Inner::Tile<V>>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] n: u32,
    ) {
        match tile {
            CubeOption::Some(tile) => MatrixStageReader::<Inner>::load_fragment(tile, frag, n),
            CubeOption::None => {
                MatrixStageReader::<Filled>::load_fragment::<E, V>(&V::from_int(0), frag, n)
            }
        }
    }
}

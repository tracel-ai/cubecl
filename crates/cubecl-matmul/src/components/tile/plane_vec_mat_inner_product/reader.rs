use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout,
    tile::{
        SharedTileConfig, StridedTile,
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
        #[comptime] config: SharedTileConfig,
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

        frag.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: SharedTileConfig,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.tile_size.n() {
            let line_container = frag.index_mut(n);
            line_container.line = Line::cast_from(tile.slice[UNIT_POS_X + n * tile.stride]);

            comptime![n += 1];
        }
    }
}

#[cube]
impl MatrixFragmentReader for MatrixStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric>(
        value: &V,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: SharedTileConfig,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.tile_size.n() {
            let line_container = frag.index_mut(n);
            line_container.line = Line::cast_from(*value);

            comptime![n += 1];
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
        #[comptime] config: SharedTileConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => MatrixStageReader::<Inner>::load_fragment(tile, frag, config),
            CubeOption::None => {
                MatrixStageReader::<Filled>::load_fragment::<E, V>(&V::from_int(0), frag, config)
            }
        }
    }
}

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout,
    tile::{
        Tile,
        plane_vec_mat_inner_product::{LineContainer, config::PlaneVecMatInnerProductConfig},
        reader::{Filled, Strided, TileKind, TileReader},
    },
};

/// Reader for the vector side of the VecMat operation
#[derive(CubeType)]
pub struct VectorTileReader {}

/// Generic matrix reader over any tile type
#[cube]
pub(super) trait MatrixFragmentReader: TileReader {
    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    );
}

/// Reader for the matrix side of the VecMat operation. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct MatrixTileReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl VectorTileReader {
    pub fn load_fragment<E: Numeric, V: Numeric>(tile: Tile<V>, frag: &mut LineContainer<E>) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        frag.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }
}

impl TileReader for VectorTileReader {
    type TileKind = Strided;
}

#[cube]
impl MatrixFragmentReader for MatrixTileReader<Strided> {
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: Tile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let line_container = frag.index_mut(n);
            line_container.line = Line::cast_from(tile.slice[UNIT_POS_X + n * tile.stride]);

            comptime![n += 1];
        }
    }
}

#[cube]
impl MatrixFragmentReader for MatrixTileReader<Filled> {
    fn load_fragment<E: Numeric, V: Numeric>(
        value: V,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let line_container = frag.index_mut(n);
            line_container.line = Line::cast_from(value);

            comptime![n += 1];
        }
    }
}

#[cube]
impl<Inner: TileKind> MatrixFragmentReader for MatrixTileReader<CubeOption<Inner>>
where
    MatrixTileReader<Inner>: MatrixFragmentReader<TileKind = Inner>,
{
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => MatrixTileReader::<Inner>::load_fragment(tile, frag, config),
            CubeOption::None => {
                MatrixTileReader::<Filled>::load_fragment::<E, V>(V::from_int(0), frag, config)
            }
        }
    }
}

impl<Kind: TileKind> TileReader for MatrixTileReader<Kind> {
    type TileKind = Kind;
}

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout,
    tile::{
        Tile,
        loader::{Filled, Strided, TileKind, TileLoader},
        plane_vec_mat_inner_product::{LineContainer, config::PlaneVecMatInnerProductConfig},
    },
};

/// Loader for the vector side of the VecMat operation
#[derive(CubeType)]
pub struct VectorTileLoader {}

/// Generic matrix loader over any tile type
#[cube]
pub(super) trait MatrixFragmentLoader: TileLoader {
    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    );
}

/// Loader for the matrix side of the VecMat operation. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct MatrixTileLoader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl VectorTileLoader {
    pub fn load_fragment<E: Numeric, V: Numeric>(tile: Tile<V>, frag: &mut LineContainer<E>) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        frag.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }
}

impl TileLoader for VectorTileLoader {
    type TileKind = Strided;
}

#[cube]
impl MatrixFragmentLoader for MatrixTileLoader<Strided> {
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
impl MatrixFragmentLoader for MatrixTileLoader<Filled> {
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
impl<Inner: TileKind> MatrixFragmentLoader for MatrixTileLoader<CubeOption<Inner>>
where
    MatrixTileLoader<Inner>: MatrixFragmentLoader<TileKind = Inner>,
{
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => MatrixTileLoader::<Inner>::load_fragment(tile, frag, config),
            CubeOption::None => {
                MatrixTileLoader::<Filled>::load_fragment::<E, V>(V::from_int(0), frag, config)
            }
        }
    }
}

impl<Kind: TileKind> TileLoader for MatrixTileLoader<Kind> {
    type TileKind = Kind;
}

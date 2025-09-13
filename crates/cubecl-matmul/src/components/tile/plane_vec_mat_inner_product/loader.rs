use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::{
    MatrixLayout,
    tile::{
        Tile,
        loader::{Filled, Loader, Strided, TileKind},
        plane_vec_mat_inner_product::{LineContainer, config::PlaneVecMatInnerProductConfig},
    },
};

#[derive(CubeType)]
pub struct VectorLoader {}

#[cube]
pub(super) trait TileMatrixLoader: Loader {
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    );
}

#[derive(CubeType)]
pub struct MatrixLoader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl VectorLoader {
    pub fn fill_fragment<E: Numeric, V: Numeric>(tile: Tile<V>, frag: &mut LineContainer<E>) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        frag.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }
}

impl Loader for VectorLoader {
    type TileKind = Strided;
}

#[cube]
impl TileMatrixLoader for MatrixLoader<Strided> {
    fn fill_fragment<E: Numeric, V: Numeric>(
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
impl TileMatrixLoader for MatrixLoader<Filled> {
    fn fill_fragment<E: Numeric, V: Numeric>(
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
impl<Inner: TileKind> TileMatrixLoader for MatrixLoader<CubeOption<Inner>>
where
    MatrixLoader<Inner>: TileMatrixLoader<TileKind = Inner>,
{
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        frag: &mut Sequence<LineContainer<E>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => MatrixLoader::<Inner>::fill_fragment(tile, frag, config),
            CubeOption::None => {
                MatrixLoader::<Filled>::fill_fragment::<E, V>(V::from_int(0), frag, config)
            }
        }
    }
}

impl<Kind: TileKind> Loader for MatrixLoader<Kind> {
    type TileKind = Kind;
}

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::{
    MatrixLayout,
    tile::{
        Tile,
        loader::{FillLoader, Loader, TileKind, TileLoader},
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
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        frag.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }
}

impl Loader for VectorLoader {
    type TileKind = TileLoader;
}

#[cube]
impl TileMatrixLoader for MatrixLoader<TileLoader> {
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
impl TileMatrixLoader for MatrixLoader<FillLoader> {
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

impl<Kind: TileKind> Loader for MatrixLoader<Kind> {
    type TileKind = Kind;
}

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::tile::{
    Tile,
    loader::{FillLoader, Loader, TileKind, TileLoader},
};

#[cube]
pub(crate) trait CmmaTileLoader: Loader {
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut cmma::Matrix<E>,
        layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] line_size: u32,
    );
}

#[derive(CubeType)]
pub struct CmmaLoader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl CmmaTileLoader for CmmaLoader<TileLoader> {
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: Tile<V>,
        fragment: &mut cmma::Matrix<E>,
        layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] line_size: u32,
    ) {
        let (slice, stride) = tile.as_unlined(line_size);
        match layout {
            CubeOption::None => cmma::load(fragment, &slice, stride),
            CubeOption::Some(layout) => cmma::load_with_layout(fragment, &slice, stride, layout),
        }
    }
}

#[cube]
impl CmmaTileLoader for CmmaLoader<FillLoader> {
    fn fill_fragment<E: Numeric, V: Numeric>(
        value: V,
        fragment: &mut cmma::Matrix<E>,
        _layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] _line_size: u32,
    ) {
        cmma::fill(fragment, E::cast_from(value));
    }
}

#[cube]
impl<Inner: TileKind> CmmaTileLoader for CmmaLoader<CubeOption<Inner>>
where
    CmmaLoader<Inner>: CmmaTileLoader<TileKind = Inner>,
{
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        fragment: &mut cmma::Matrix<E>,
        layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] line_size: u32,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                CmmaLoader::<Inner>::fill_fragment(tile, fragment, layout, line_size)
            }
            CubeOption::None => CmmaLoader::<FillLoader>::fill_fragment::<E, V>(
                V::from_int(0),
                fragment,
                layout,
                line_size,
            ),
        }
    }
}

impl<Kind: TileKind> Loader for CmmaLoader<Kind> {
    type TileKind = Kind;
}

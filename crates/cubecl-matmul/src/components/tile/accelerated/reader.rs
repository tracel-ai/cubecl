use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::tile::{
    Tile,
    reader::{Filled, Strided, TileKind, TileReader},
};

/// Generic CMMA reader over any tile type
#[cube]
pub(crate) trait CmmaFragmentReader: TileReader {
    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut cmma::Matrix<E>,
        layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] line_size: u32,
    );
}

/// Reader using the cmma load/fill functions. Tile kind determines implementation.
#[derive(CubeType)]
pub struct CmmaTileReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
impl CmmaFragmentReader for CmmaTileReader<Strided> {
    fn load_fragment<E: Numeric, V: Numeric>(
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
impl CmmaFragmentReader for CmmaTileReader<Filled> {
    fn load_fragment<E: Numeric, V: Numeric>(
        value: V,
        fragment: &mut cmma::Matrix<E>,
        _layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] _line_size: u32,
    ) {
        cmma::fill(fragment, E::cast_from(value));
    }
}

#[cube]
impl<Inner: TileKind> CmmaFragmentReader for CmmaTileReader<CubeOption<Inner>>
where
    CmmaTileReader<Inner>: CmmaFragmentReader<TileKind = Inner>,
{
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        fragment: &mut cmma::Matrix<E>,
        layout: CubeOption<cmma::MatrixLayout>,
        #[comptime] line_size: u32,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                CmmaTileReader::<Inner>::load_fragment(tile, fragment, layout, line_size)
            }
            CubeOption::None => CmmaTileReader::<Filled>::load_fragment::<E, V>(
                V::from_int(0),
                fragment,
                layout,
                line_size,
            ),
        }
    }
}

impl<Kind: TileKind> TileReader for CmmaTileReader<Kind> {
    type TileKind = Kind;
}

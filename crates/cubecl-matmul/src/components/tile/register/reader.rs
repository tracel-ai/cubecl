use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::{
    MatrixLayout, StageIdent,
    tile::{
        StridedTile,
        io::{Filled, Strided, TileKind},
        register::{
            RegisterMatmul, UnitFragment,
            config::{ProductType, RegisterMatmulConfig},
        },
    },
};

/// Reader for the register matmul fragments. Implementation depends on the tile kind.
#[derive(CubeType)]
pub struct RegisterStageReader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

/// Generic register reader over any tile kind
#[cube]
pub(super) trait RegisterFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &<Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    );
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        frag: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        // Could these be unified somehow?
        match ident {
            StageIdent::Lhs => load_lhs(tile, frag, config),
            StageIdent::Rhs => load_rhs(tile, frag, config),
            StageIdent::Acc => load_acc(tile, frag, config),
            StageIdent::Out => panic!("Can't load out"),
        }
    }
}

type MM = RegisterMatmul<Strided>;

#[cube]
fn load_lhs<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match config.product_type {
        ProductType::Inner => match comptime!(frag.layout) {
            MatrixLayout::RowMajor => {
                MM::load_plain(tile, &mut frag.array, size.m(), size.k(), tile.line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_transposed(tile, &mut frag.array, size.k(), size.m(), tile.line_size);
            }
        },
        ProductType::Outer => match comptime!(frag.layout) {
            MatrixLayout::RowMajor => {
                MM::load_transposed(tile, &mut frag.array, size.m(), size.k(), tile.line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_plain(tile, &mut frag.array, size.k(), size.m(), tile.line_size);
            }
        },
    }
}

#[cube]
fn load_rhs<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match config.product_type {
        ProductType::Inner => match comptime!(frag.layout) {
            MatrixLayout::RowMajor => {
                MM::load_transposed(tile, &mut frag.array, size.k(), size.n(), tile.line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_plain(tile, &mut frag.array, size.n(), size.k(), tile.line_size);
            }
        },
        ProductType::Outer => match comptime!(frag.layout) {
            MatrixLayout::RowMajor => {
                MM::load_plain(tile, &mut frag.array, size.k(), size.n(), tile.line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_transposed(tile, &mut frag.array, size.n(), size.k(), tile.line_size);
            }
        },
    }
}

#[cube]
fn load_acc<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut UnitFragment<E>,
    #[comptime] config: RegisterMatmulConfig,
) {
    let size = config.shared.tile_size;

    match comptime!(frag.layout) {
        MatrixLayout::RowMajor => {
            MM::load_plain(tile, &mut frag.array, size.m(), size.n(), tile.line_size);
        }
        MatrixLayout::ColMajor => {
            MM::load_transposed(tile, &mut frag.array, size.n(), size.m(), tile.line_size);
        }
    }
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric>(
        value: &V,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        let size = config.shared.tile_size;
        let size = match ident {
            StageIdent::Lhs => size.mk(),
            StageIdent::Rhs => size.nk(),
            StageIdent::Acc => size.mn(),
            StageIdent::Out => size.mn(),
        };

        for i in 0..size {
            fragment.array[i] = E::cast_from(*value);
        }
    }
}

#[cube]
impl<Inner: TileKind> RegisterFragmentReader for RegisterStageReader<CubeOption<Inner>>
where
    RegisterStageReader<Inner>: RegisterFragmentReader<TileKind = Inner>,
{
    type TileKind = CubeOption<Inner>;

    fn load_fragment<E: Numeric, V: Numeric>(
        tile: &CubeOption<Inner::Tile<V>>,
        fragment: &mut UnitFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                RegisterStageReader::<Inner>::load_fragment(tile, fragment, ident, config)
            }
            CubeOption::None => RegisterStageReader::<Filled>::load_fragment::<E, V>(
                &V::from_int(0),
                fragment,
                ident,
                config,
            ),
        }
    }
}

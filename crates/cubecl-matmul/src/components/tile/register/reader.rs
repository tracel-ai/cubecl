use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::{
    MatrixLayout, StageIdent,
    tile::{
        StridedTile, TileConfig,
        io::{Filled, StageReader, Strided, TileKind},
        register::{
            RegisterMatmul,
            config::{ProductType, RegisterConfig},
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
pub(super) trait RegisterFragmentReader: StageReader {
    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    );
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Strided> {
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: StridedTile<V>,
        frag: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    ) {
        // Could these be unified somehow?
        match ident {
            StageIdent::Lhs => load_lhs(&tile, frag, config),
            StageIdent::Rhs => load_rhs(&tile, frag, config),
            StageIdent::Acc => load_acc(&tile, frag, config),
        }
    }
}

type MM = RegisterMatmul<Strided>;

#[cube]
fn load_lhs<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut Array<E>,
    #[comptime] config: RegisterConfig,
) {
    let size = config.tile_size();
    let line_size = config.stage_line_size(StageIdent::Lhs);
    let layout = config.matrix_layout(StageIdent::Lhs);

    match config.product_type() {
        ProductType::Inner => match layout {
            MatrixLayout::RowMajor => {
                MM::load_plain(tile, frag, size.m(), size.k(), line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_transposed(tile, frag, size.k(), size.m(), line_size);
            }
        },
        ProductType::Outer => match layout {
            MatrixLayout::RowMajor => {
                MM::load_transposed(tile, frag, size.m(), size.k(), line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_plain(tile, frag, size.k(), size.m(), line_size);
            }
        },
    }
}

#[cube]
fn load_rhs<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut Array<E>,
    #[comptime] config: RegisterConfig,
) {
    let size = config.tile_size();
    let line_size = config.stage_line_size(StageIdent::Rhs);
    let layout = config.matrix_layout(StageIdent::Rhs);

    match config.product_type() {
        ProductType::Inner => match layout {
            MatrixLayout::RowMajor => {
                MM::load_transposed(tile, frag, size.k(), size.n(), line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_plain(tile, frag, size.n(), size.k(), line_size);
            }
        },
        ProductType::Outer => match layout {
            MatrixLayout::RowMajor => {
                MM::load_plain(tile, frag, size.k(), size.n(), line_size);
            }
            MatrixLayout::ColMajor => {
                MM::load_transposed(tile, frag, size.n(), size.k(), line_size);
            }
        },
    }
}

#[cube]
fn load_acc<E: Numeric, V: Numeric>(
    tile: &StridedTile<V>,
    frag: &mut Array<E>,
    #[comptime] config: RegisterConfig,
) {
    let size = config.tile_size();
    let line_size = config.stage_line_size(StageIdent::Acc);
    let layout = config.matrix_layout(StageIdent::Acc);

    match layout {
        MatrixLayout::RowMajor => {
            MM::load_plain(tile, frag, size.m(), size.n(), line_size);
        }
        MatrixLayout::ColMajor => {
            MM::load_transposed(tile, frag, size.n(), size.m(), line_size);
        }
    }
}

#[cube]
impl RegisterFragmentReader for RegisterStageReader<Filled> {
    fn load_fragment<E: Numeric, V: Numeric>(
        value: V,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    ) {
        let size = config.tile_size();
        let size = match ident {
            StageIdent::Lhs => size.mk(),
            StageIdent::Rhs => size.nk(),
            StageIdent::Acc => size.mn(),
        };

        for i in 0..size {
            fragment[i] = E::cast_from(value);
        }
    }
}

#[cube]
impl<Inner: TileKind> RegisterFragmentReader for RegisterStageReader<CubeOption<Inner>>
where
    RegisterStageReader<Inner>: RegisterFragmentReader<TileKind = Inner>,
{
    fn load_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                RegisterStageReader::<Inner>::load_fragment(tile, fragment, ident, config)
            }
            CubeOption::None => RegisterStageReader::<Filled>::load_fragment::<E, V>(
                V::from_int(0),
                fragment,
                ident,
                config,
            ),
        }
    }
}

impl<Kind: TileKind> StageReader for RegisterStageReader<Kind> {
    type TileKind = Kind;
}

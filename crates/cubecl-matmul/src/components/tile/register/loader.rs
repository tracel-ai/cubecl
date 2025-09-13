use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::components::{
    MatrixLayout, StageIdent,
    tile::{
        Tile, TileConfig,
        loader::{Filled, Loader, Strided, TileKind},
        register::{
            RegisterMatmul,
            config::{ProductType, RegisterConfig},
        },
    },
};

#[derive(CubeType)]
pub struct RegisterLoader<Kind: TileKind> {
    #[cube(comptime)]
    _ty: PhantomData<Kind>,
}

#[cube]
pub(super) trait TileRegisterLoader: Loader {
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: <Self::TileKind as TileKind>::Tile<V>,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    );
}

#[cube]
impl TileRegisterLoader for RegisterLoader<Strided> {
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: Tile<V>,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    ) {
        let size = config.tile_size();
        let line_size = config.stage_line_size(ident);
        let layout = config.matrix_layout(ident);

        let (row, col) = match ident {
            StageIdent::Lhs => (size.m(), size.k()),
            StageIdent::Rhs => (size.k(), size.n()),
            StageIdent::Acc => (size.m(), size.n()),
        };

        match config.product_type() {
            ProductType::Inner => match layout {
                MatrixLayout::RowMajor => {
                    RegisterMatmul::<Strided>::fill_transposed(
                        &tile, fragment, row, col, line_size,
                    );
                }
                MatrixLayout::ColMajor => {
                    RegisterMatmul::<Strided>::fill_plain(&tile, fragment, col, row, line_size);
                }
            },
            ProductType::Outer => match layout {
                MatrixLayout::RowMajor => {
                    RegisterMatmul::<Strided>::fill_plain(&tile, fragment, row, col, line_size);
                }
                MatrixLayout::ColMajor => {
                    RegisterMatmul::<Strided>::fill_transposed(
                        &tile, fragment, col, row, line_size,
                    );
                }
            },
        }
    }
}

#[cube]
impl TileRegisterLoader for RegisterLoader<Filled> {
    fn fill_fragment<E: Numeric, V: Numeric>(
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
impl<Inner: TileKind> TileRegisterLoader for RegisterLoader<CubeOption<Inner>>
where
    RegisterLoader<Inner>: TileRegisterLoader<TileKind = Inner>,
{
    fn fill_fragment<E: Numeric, V: Numeric>(
        tile: CubeOption<Inner::Tile<V>>,
        fragment: &mut Array<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: RegisterConfig,
    ) {
        match tile {
            CubeOption::Some(tile) => {
                RegisterLoader::<Inner>::fill_fragment(tile, fragment, ident, config)
            }
            CubeOption::None => RegisterLoader::<Filled>::fill_fragment::<E, V>(
                V::from_int(0),
                fragment,
                ident,
                config,
            ),
        }
    }
}

impl<Kind: TileKind> Loader for RegisterLoader<Kind> {
    type TileKind = Kind;
}

use std::marker::PhantomData;

use crate::components::tile::{
    SharedTileConfig, TileMatmul, cmma::reader::CmmaFragmentReader, io::Filled,
};
use crate::components::tile::{
    cmma::reader::CmmaStageReader,
    io::{Strided, TileKind},
};
use crate::components::tile::{cmma::writer::CmmaStageWriter, tile_data::StridedTile};
use crate::components::{MatrixLayout, as_cmma_layout};
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use cubecl_std::CubeOption;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul<Acc: TileKind = Filled> {
    _ty: PhantomData<Acc>,
}

#[derive(CubeType)]
pub struct Fragment<E: Numeric> {
    fragment: cmma::Matrix<E>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, AccTile: TileKind> TileMatmul<L, R, A>
    for CmmaMatmul<AccTile>
where
    CmmaStageReader<AccTile>: CmmaFragmentReader<TileKind = AccTile>,
{
    type Config = SharedTileConfig;

    type LhsFragment = Fragment<L>;
    type RhsFragment = Fragment<R>;
    type AccFragment = Fragment<A>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<L, R, A, A>(&lhs.fragment, &rhs.fragment, &out.fragment, &out.fragment);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let size = config.tile_size;

        Fragment::<L> {
            fragment: unsafe {
                cmma::Matrix::<L>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m(),
                    size.n(),
                    size.k(),
                    as_cmma_layout(layout),
                )
            },
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let size = config.tile_size;

        Fragment::<R> {
            fragment: unsafe {
                cmma::Matrix::uninitialized(
                    cmma::MatrixIdent::B,
                    size.m(),
                    size.n(),
                    size.k(),
                    as_cmma_layout(layout),
                )
            },
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let size = config.tile_size;

        Fragment::<A> {
            fragment: unsafe {
                cmma::Matrix::<A>::uninitialized(
                    cmma::MatrixIdent::Accumulator,
                    size.m(),
                    size.n(),
                    size.k(),
                    cmma::MatrixLayout::Undefined,
                )
            },
            layout,
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Self::LhsTile>::load_fragment(
            tile,
            &mut lhs.fragment,
            CubeOption::new_None(),
        );
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Self::RhsTile>::load_fragment(
            tile,
            &mut rhs.fragment,
            CubeOption::new_None(),
        );
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Self::AccTile>::load_fragment(
            tile,
            &mut acc.fragment,
            CubeOption::new_Some(as_cmma_layout(acc.layout)),
        );
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        out: &Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        let out = cmma::cast::<A, E>(&out.fragment);
        CmmaStageWriter::store_fragment(tile, &out);
    }
}

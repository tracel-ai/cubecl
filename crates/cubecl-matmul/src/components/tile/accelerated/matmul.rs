use std::marker::PhantomData;

use crate::components::tile::{TileConfig, TileMatmul, accelerated::reader::CmmaFragmentReader};
use crate::components::tile::{accelerated::writer::CmmaStageWriter, tile_data::StridedTile};
use crate::components::tile::{
    accelerated::{config::AcceleratedConfig, reader::CmmaStageReader},
    io::{Strided, TileKind},
};
use crate::components::{StageIdent, as_cmma_layout};
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use cubecl_std::CubeOption;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct AcceleratedMatmul<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, AccTile: TileKind> TileMatmul<L, R, A>
    for AcceleratedMatmul<AccTile>
where
    CmmaStageReader<AccTile>: CmmaFragmentReader<TileKind = AccTile>,
{
    type Config = AcceleratedConfig;
    type LhsFragment = cmma::Matrix<L>;
    type RhsFragment = cmma::Matrix<R>;
    type AccFragment = cmma::Matrix<A>;

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
        cmma::execute::<L, R, A, A>(lhs, rhs, out, out);
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::LhsFragment {
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Lhs);
        unsafe {
            cmma::Matrix::<L>::uninitialized(
                cmma::MatrixIdent::A,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::RhsFragment {
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Rhs);
        unsafe {
            cmma::Matrix::<R>::uninitialized(
                cmma::MatrixIdent::B,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Self::LhsTile>::load_fragment(tile, lhs, CubeOption::new_None());
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        CmmaStageReader::<Self::RhsTile>::load_fragment(tile, rhs, CubeOption::new_None());
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        CmmaStageReader::<Self::AccTile>::load_fragment(tile, acc, CubeOption::new_Some(layout));
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        out: &Self::AccFragment,
        #[comptime] _config: Self::Config,
    ) {
        let out = cmma::cast::<A, E>(out);
        CmmaStageWriter::store_fragment(tile, &out);
    }

    fn allocate_acc(#[comptime] config: Self::Config) -> Self::AccFragment {
        let size = config.tile_size();
        unsafe {
            cmma::Matrix::<A>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }
}

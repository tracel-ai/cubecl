use std::marker::PhantomData;

use crate::components::tile::tile_data::StridedTile;
use crate::components::tile::{TileConfig, TileMatmul, accelerated::reader::CmmaFragmentReader};
use crate::components::tile::{
    accelerated::{config::AcceleratedConfig, reader::CmmaTileReader},
    reader::{Strided, TileKind},
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
impl<L: Numeric, R: Numeric, A: Numeric, Acc: TileKind> TileMatmul<L, R, A>
    for AcceleratedMatmul<Acc>
where
    CmmaTileReader<Acc>: CmmaFragmentReader<TileKind = Acc>,
{
    type Config = AcceleratedConfig;
    type LhsFragment = cmma::Matrix<L>;
    type RhsFragment = cmma::Matrix<R>;
    type AccFragment = cmma::Matrix<A>;

    type LhsTileReader = CmmaTileReader<Strided>;
    type RhsTileReader = CmmaTileReader<Strided>;
    type AccTileReader = CmmaTileReader<Acc>;

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
        tile: StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::LhsTileReader::load_fragment(
            tile,
            lhs,
            CubeOption::new_None(),
            config.stage_line_size(StageIdent::Lhs),
        );
    }

    fn load_rhs<E: Numeric>(
        tile: StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::RhsTileReader::load_fragment(
            tile,
            rhs,
            CubeOption::new_None(),
            config.stage_line_size(StageIdent::Rhs),
        );
    }

    fn load_acc<E: Numeric>(
        tile: Acc::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        Self::AccTileReader::load_fragment(
            tile,
            acc,
            CubeOption::new_Some(layout),
            config.stage_line_size(StageIdent::Acc),
        );
    }

    fn write_results<E: Numeric>(
        out: &Self::AccFragment,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<A, E>(out);
        cmma::store(
            slice,
            &acc,
            config.tile_size().n(),
            cmma::MatrixLayout::RowMajor,
        );
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

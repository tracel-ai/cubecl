use std::marker::PhantomData;

use crate::components::tile::tile_data::Tile;
use crate::components::tile::{TileConfig, TileMatmul, accelerated::loader::CmmaTileLoader};
use crate::components::tile::{
    accelerated::{config::AcceleratedConfig, loader::CmmaLoader},
    loader::{Strided, TileKind},
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
    CmmaLoader<Acc>: CmmaTileLoader<TileKind = Acc>,
{
    type Config = AcceleratedConfig;
    type Lhs = cmma::Matrix<L>;
    type Rhs = cmma::Matrix<R>;
    type Accumulator = cmma::Matrix<A>;

    type LhsLoader = CmmaLoader<Strided>;
    type RhsLoader = CmmaLoader<Strided>;
    type AccLoader = CmmaLoader<Acc>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<L, R, A, A>(lhs, rhs, out, out);
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
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

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
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

    fn fill_lhs<E: Numeric>(tile: Tile<E>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
        Self::LhsLoader::fill_fragment(
            tile,
            lhs,
            CubeOption::new_None(),
            config.stage_line_size(StageIdent::Lhs),
        );
    }

    fn fill_rhs<E: Numeric>(tile: Tile<E>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        Self::RhsLoader::fill_fragment(
            tile,
            rhs,
            CubeOption::new_None(),
            config.stage_line_size(StageIdent::Rhs),
        );
    }

    fn fill_acc<E: Numeric>(
        tile: Acc::Tile<E>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        Self::AccLoader::fill_fragment(
            tile,
            acc,
            CubeOption::new_Some(layout),
            config.stage_line_size(StageIdent::Acc),
        );
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
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

    fn allocate_acc(#[comptime] config: Self::Config) -> Self::Accumulator {
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

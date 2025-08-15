use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::tile_data::Tile;
use crate::components::tile::{TileConfig, TileMatmul};
use crate::components::{StageIdent, as_cmma_layout};
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct AcceleratedMatmul;

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for AcceleratedMatmul {
    type Config = AcceleratedConfig;
    type Lhs = cmma::Matrix<L>;
    type Rhs = cmma::Matrix<R>;
    type Accumulator = cmma::Matrix<A>;

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

    fn fill_lhs<E: Numeric>(tile: &Tile<E>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Lhs, config);
        cmma::load(lhs, &slice, stride);
    }

    fn fill_rhs<E: Numeric>(tile: &Tile<E>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Rhs, config);
        cmma::load(rhs, &slice, stride);
    }

    fn fill_accumulator(
        tile: &Tile<A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Acc, config);
        cmma::load_with_layout(acc, &slice, stride, layout);
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

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
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

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, A::from_int(0));
    }
}

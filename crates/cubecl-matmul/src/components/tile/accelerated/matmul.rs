use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::tile_data::Tile;
use crate::components::tile::{TileConfig, TileMatmul};
use crate::components::{MatmulPrecision, StageIdent, as_cmma_layout};
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct AcceleratedMatmul;

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for AcceleratedMatmul {
    type Config = AcceleratedConfig;
    type Lhs = cmma::Matrix<MP::ES>;
    type Rhs = cmma::Matrix<MP::ES>;
    type Accumulator = cmma::Matrix<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<MP::ES, MP::ES, MP::EA, MP::EA>(lhs, rhs, out, out);
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        let size = config.tile_size();
        let layout = config.matrix_layout(StageIdent::Lhs);
        unsafe {
            cmma::Matrix::<MP::ES>::uninitialized(
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
            cmma::Matrix::<MP::ES>::uninitialized(
                cmma::MatrixIdent::B,
                size.m(),
                size.n(),
                size.k(),
                as_cmma_layout(layout),
            )
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Lhs, config);
        cmma::load(lhs, &slice, stride);
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Rhs, config);
        cmma::load(rhs, &slice, stride);
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let layout = comptime!(as_cmma_layout(config.matrix_layout(StageIdent::Acc)));
        let (slice, stride) = tile.as_unlined::<Self::Config>(StageIdent::Acc, config);
        cmma::load_with_layout(acc, &slice, stride, layout);
    }

    fn write_results(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<MP::EA, MP::EO>(out);
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
            cmma::Matrix::<MP::EA>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, MP::EA::from_int(0));
    }
}

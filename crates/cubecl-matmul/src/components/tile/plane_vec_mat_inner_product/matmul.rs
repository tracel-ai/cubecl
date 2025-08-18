use crate::components::tile::TileMatmul;
use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::tile_data::Tile;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct;

/// Doesn't impact performance much, but may increase kernel size too much when true (often ~6X).
///
/// TODO: make it configurable
static UNROLL: bool = false;

#[derive(CubeType)]
/// Contains the accumulated result in a row-major array of size rows x cols
pub struct TileAccumulator<EA: Numeric> {
    data: Array<EA>,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    cols: u32,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for PlaneVecMatInnerProduct {
    type Config = PlaneVecMatInnerProductConfig;
    type Lhs = Array<L>;
    type Rhs = Array<R>;
    type Accumulator = TileAccumulator<A>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        comptime!(todo!())
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        comptime!(todo!())
    }

    fn fill_lhs<E: Numeric>(tile: &Tile<E>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
    }

    fn fill_rhs<E: Numeric>(tile: &Tile<E>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
    }

    fn fill_accumulator(
        tile: &Tile<A>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
    }

    fn write_results<E: Numeric>(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        comptime!(todo!())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {}
}

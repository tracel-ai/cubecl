use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::tile_data::Tile;
use crate::components::tile::{TileConfig, TileMatmul};
use crate::components::{MatrixLayout, StageIdent};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct;

#[derive(CubeType)]
pub struct LineContainer<E: Numeric> {
    line: Line<E>,
}

#[cube]
impl<E: Numeric> LineContainer<E> {
    fn new(#[comptime] size: u32) -> LineContainer<E> {
        LineContainer::<E> {
            line: Line::empty(size),
        }
    }
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for PlaneVecMatInnerProduct {
    type Config = PlaneVecMatInnerProductConfig;

    // One line per unit in the plane
    type Lhs = LineContainer<L>;
    // For each n: one line per unit in the plane
    type Rhs = Sequence<LineContainer<R>>;

    // For each n: one line stored at unit pos 0, that will be reduced to a scalar only when writing at the end
    type Accumulator = Sequence<LineContainer<A>>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let lhs: Line<A> = Line::cast_from(lhs.line);
            let rhs: Line<A> = Line::cast_from(rhs.index(n).line);

            plane_sum_lined(lhs * rhs, acc.index_mut(n), config.reduce_line_size());

            comptime![n += 1];
        }
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        LineContainer::<L>::new(config.reduce_line_size())
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.n() {
            rhs.push(LineContainer::new(config.reduce_line_size()))
        }
        rhs
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.n() {
            acc.push(LineContainer::new(config.reduce_line_size()))
        }
        acc
    }

    fn fill_lhs<E: Numeric>(
        tile: &Tile<E>,
        lhs: &mut Self::Lhs,
        #[comptime] _config: Self::Config,
    ) {
        comptime!(assert!(tile.layout == MatrixLayout::RowMajor));

        lhs.line = Line::cast_from(tile.slice[UNIT_POS_X]);
    }

    fn fill_rhs<E: Numeric>(tile: &Tile<E>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        comptime!(assert!(tile.layout == MatrixLayout::ColMajor));

        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let line_container = rhs.index_mut(n);
            line_container.line = Line::cast_from(tile.slice[UNIT_POS_X + n * tile.stride]);

            comptime![n += 1];
        }
    }

    fn fill_accumulator(
        tile: &Tile<A>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let line_container = acc.index_mut(n);
            line_container.line = Line::cast_from(tile.slice[UNIT_POS_X + n * tile.stride]);

            comptime![n += 1];
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let line_container = acc.index_mut(n);
            line_container.line = Line::cast_from(0);

            comptime![n += 1];
        }
    }

    fn write_results<E: Numeric>(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let out_line_size = config.stage_line_size(StageIdent::Acc);
            let total_out_lines = config.n() / out_line_size;
            let mut out_line_iter = comptime![0];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..total_out_lines {
                let mut out_line = Line::<E>::empty(out_line_size);
                let mut within_line = comptime![0];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..out_line_size {
                    let n = comptime!(out_line_iter * out_line_size + within_line);

                    let line_container = acc.index(n);
                    let mut sum = A::from_int(0);
                    for i in 0..config.reduce_line_size() {
                        sum += line_container.line[i];
                    }

                    out_line[within_line] = E::cast_from(sum);
                    comptime![within_line += 1];
                }

                slice[out_line_iter] = out_line;
                comptime![out_line_iter += 1];
            }
        }
    }
}

#[cube]
fn plane_sum_lined<E: Numeric>(
    line_to_sum: Line<E>,
    line_accumulator: &mut LineContainer<E>,
    #[comptime] line_size: u32,
) {
    let mut line_iterator = comptime![0];

    #[unroll]
    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..line_size {
        line_accumulator.line[line_iterator] += plane_sum(line_to_sum[line_iterator]);

        comptime![line_iterator += 1];
    }
}

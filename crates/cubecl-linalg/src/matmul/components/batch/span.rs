use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::{self};

use super::shared::gmm_execute;

#[derive(CubeType)]
/// Area of a tensor a cube is responsible of performing matmul
/// Similar to the concept of tensor slice, but specialized for matmul constraints
pub struct Span {
    row: SpanDim,
    col: SpanDim,
    batch: SpanDim,
}

#[derive(CubeType)]
/// Span information in one dimension
pub struct SpanDim {
    start: u32,
    end: u32,
    step: u32,
}

#[cube]
/// Iterates on several global matmul across a span
pub trait SpanMatmul: 'static + Send + Sync {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    );
}

#[derive(CubeType)]
/// Iterates on global matmuls in a row major fashion
pub struct RowMajorSpanMatmul {}

#[derive(CubeType)]
/// Iterates on global matmuls in a col major fashion
pub struct ColMajorSpanMatmul {}

#[derive(CubeType)]
/// Iterates on global matmuls following the swizzle algorithm
///
/// The swizzle algorithm processes  W elements per row in a top-down pass,
/// then shifts to the next W columns in a bottom-up pass.
/// This zigzag (top-down, bottom-up) repeats, covering the matrix span by span.
pub struct SwizzleSpanMatmul<const W: u32> {}

#[cube]
impl Span {
    pub fn new(row: SpanDim, col: SpanDim, batch: SpanDim) -> Span {
        Span { row, col, batch }
    }
}

#[cube]
impl SpanDim {
    pub fn new(shape: u32, stage: u32, cube_pos: u32, num_cubes: u32) -> SpanDim {
        let num_stages = (shape + stage - 1) / stage;
        let num = (num_stages + num_cubes - 1) / num_cubes;
        let span = num * stage;
        let start = cube_pos * span;
        let end = Min::min(start + span, shape);
        SpanDim {
            start,
            end,
            step: stage,
        }
    }

    pub fn num_iterations(&self) -> u32 {
        let range = self.end - self.start;
        (range + self.step - 1) / self.step
    }
}

#[cube]
impl SpanMatmul for RowMajorSpanMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            for row_iter in range_stepped(span.row.start, span.row.end, span.row.step) {
                for col_iter in range_stepped(span.col.start, span.col.end, span.col.step) {
                    GMM::reset_accumulator(&mut acc, config);
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, row_iter, col_iter, batch_iter, &mut acc, k_range, config,
                    );
                }
            }
        }
    }
}
#[cube]
impl SpanMatmul for ColMajorSpanMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            for col_iter in range_stepped(span.col.start, span.col.end, span.col.step) {
                for row_iter in range_stepped(span.row.start, span.row.end, span.row.step) {
                    GMM::reset_accumulator(&mut acc, config);
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, row_iter, col_iter, batch_iter, &mut acc, k_range, config,
                    );
                }
            }
        }
    }
}

#[cube]
impl<const W: u32> SpanMatmul for SwizzleSpanMatmul<W> {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        let num_swizzle = span.row.num_iterations() * span.col.num_iterations();

        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            for n in 0..num_swizzle {
                GMM::reset_accumulator(&mut acc, config);
                let (row, col) = swizzle(n, span.row.num_iterations(), W);

                let row_iter = span.row.start + row * span.row.step;
                let col_iter = span.col.start + col * span.col.step;
                gmm_execute::<EG, ES, GMM>(
                    lhs, rhs, out, row_iter, col_iter, batch_iter, &mut acc, k_range, config,
                );
            }
        }
    }
}

#[cube]
pub fn swizzle(nth: u32, height: u32, #[comptime] swizzle_width: u32) -> (u32, u32) {
    let num_elem_per_swizzle_col = height * swizzle_width;

    let swizzle_id = nth % num_elem_per_swizzle_col;
    let swizzle_col = nth / num_elem_per_swizzle_col;

    let col_within_swizzle = swizzle_id / height;
    let col = swizzle_col * swizzle_width + col_within_swizzle;

    let topdown_row = swizzle_id % height;
    let is_bottom_up = swizzle_col % 2;

    let row = topdown_row + is_bottom_up * (height - 2 * topdown_row - 1);

    (row, col)
}

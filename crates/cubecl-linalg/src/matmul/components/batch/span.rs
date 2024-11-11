use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global;

use super::shared::gmm_execute;

#[derive(CubeType)]
pub struct Span {
    x: SpanDim,
    y: SpanDim,
    z: SpanDim,
}

#[cube]
impl Span {
    pub fn new(x: SpanDim, y: SpanDim, z: SpanDim) -> Span {
        Span { x, y, z }
    }
}

#[derive(CubeType)]
pub struct SpanDim {
    start: u32,
    end: u32,
    step: u32,
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
pub trait SpanMatmul: 'static + Send + Sync {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    );
}

#[derive(CubeType)]
pub struct RowMajorSpanMatmul {}

#[cube]
impl SpanMatmul for RowMajorSpanMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for z_iter in range_stepped(span.z.start, span.z.end, span.z.step) {
            for x_iter in range_stepped(span.x.start, span.x.end, span.x.step) {
                for y_iter in range_stepped(span.y.start, span.y.end, span.y.step) {
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, x_iter, y_iter, z_iter, k_range, config,
                    );
                }
            }
        }
    }
}

#[derive(CubeType)]
pub struct ColMajorSpanMatmul {}

#[cube]
impl SpanMatmul for ColMajorSpanMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for z_iter in range_stepped(span.z.start, span.z.end, span.z.step) {
            for y_iter in range_stepped(span.y.start, span.y.end, span.y.step) {
                for x_iter in range_stepped(span.x.start, span.x.end, span.x.step) {
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, x_iter, y_iter, z_iter, k_range, config,
                    );
                }
            }
        }
    }
}

#[derive(CubeType)]
pub struct SwizzleSpanMatmul<const W: u32> {}

#[cube]
impl<const W: u32> SpanMatmul for SwizzleSpanMatmul<W> {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        span: Span,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        let num_swizzle = span.x.num_iterations() * span.y.num_iterations();

        for z_iter in range_stepped(span.z.start, span.z.end, span.z.step) {
            for n in 0..num_swizzle {
                let (row, col) = swizzle(n, span.x.num_iterations(), W);

                let x_iter = span.x.start + row * span.x.step;
                let y_iter = span.y.start + col * span.y.step;
                gmm_execute::<EG, ES, GMM>(lhs, rhs, out, x_iter, y_iter, z_iter, k_range, config);
            }
        }
    }
}

#[cube]
fn swizzle(nth: u32, height: u32, #[comptime] swizzle_width: u32) -> (u32, u32) {
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

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::{self, Accumulator};

use super::shared::gmm_execute;

#[derive(CubeType)]
pub struct Slice {
    x: SliceDim,
    y: SliceDim,
    z: SliceDim,
}

#[cube]
impl Slice {
    pub fn new(x: SliceDim, y: SliceDim, z: SliceDim) -> Slice {
        Slice { x, y, z }
    }
}

#[derive(CubeType)]
pub struct SliceDim {
    start: u32,
    end: u32,
    step: u32,
}

#[cube]
impl SliceDim {
    pub fn new(shape: u32, stage: u32, cube_pos: u32, num_cubes: u32) -> SliceDim {
        let num_stages = (shape + stage - 1) / stage;
        let num = (num_stages + num_cubes - 1) / num_cubes;
        let span = num * stage;
        let start = cube_pos * span;
        let end = Min::min(start + span, shape);
        SliceDim {
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
pub trait SliceMatmul: 'static + Send + Sync {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        slice: Slice,
        acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    );
}

#[derive(CubeType)]
pub struct RowMajorSliceMatmul {}

#[cube]
impl SliceMatmul for RowMajorSliceMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        slice: Slice,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for z_iter in range_stepped(slice.z.start, slice.z.end, slice.z.step) {
            for x_iter in range_stepped(slice.x.start, slice.x.end, slice.x.step) {
                for y_iter in range_stepped(slice.y.start, slice.y.end, slice.y.step) {
                    GMM::Accumulator::reset(&mut acc);
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, x_iter, y_iter, z_iter, &mut acc, k_range, config,
                    );
                }
            }
        }
    }
}

#[derive(CubeType)]
pub struct ColMajorSliceMatmul {}

#[cube]
impl SliceMatmul for ColMajorSliceMatmul {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        slice: Slice,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        for z_iter in range_stepped(slice.z.start, slice.z.end, slice.z.step) {
            for y_iter in range_stepped(slice.y.start, slice.y.end, slice.y.step) {
                for x_iter in range_stepped(slice.x.start, slice.x.end, slice.x.step) {
                    GMM::Accumulator::reset(&mut acc);
                    gmm_execute::<EG, ES, GMM>(
                        lhs, rhs, out, x_iter, y_iter, z_iter, &mut acc, k_range, config,
                    );
                }
            }
        }
    }
}

#[derive(CubeType)]
pub struct SwizzleSliceMatmul<const W: u32> {}

#[cube]
impl<const W: u32> SliceMatmul for SwizzleSliceMatmul<W> {
    fn execute<EG: Numeric, ES: Numeric, GMM: global::Matmul<EG, ES>>(
        lhs: &Tensor<Line<EG>>,
        rhs: &Tensor<Line<EG>>,
        out: &mut Tensor<Line<EG>>,
        slice: Slice,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        let num_swizzle = slice.x.num_iterations() * slice.y.num_iterations();

        for z_iter in range_stepped(slice.z.start, slice.z.end, slice.z.step) {
            for n in 0..num_swizzle {
                GMM::Accumulator::reset(&mut acc);
                let (row, col) = swizzle(n, slice.x.num_iterations(), W);

                let x_iter = slice.x.start + row * slice.x.step;
                let y_iter = slice.y.start + col * slice.y.step;
                gmm_execute::<EG, ES, GMM>(
                    lhs, rhs, out, x_iter, y_iter, z_iter, &mut acc, k_range, config,
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

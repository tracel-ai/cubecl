use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    MatmulPrecision,
    batch::shared::swizzle,
    global::{self, Quantization},
};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::shared::gmm_execute;

#[derive(CubeType)]
/// Area of a tensor a cube is responsible of performing matmul
/// Similar to the concept of tensor slice, but specialized for matmul constraints
pub struct PartitionSpan {
    row: PartitionSpanDim,
    col: PartitionSpanDim,
    batch: PartitionSpanDim,
}

#[derive(CubeType)]
pub struct PartitionSpanDim {
    start: u32,
    end: u32,
    #[cube(comptime)]
    step: u32,
    #[cube(comptime)]
    num_steps: u32,
}

#[cube]
/// Iterates on several global matmul across a span
pub trait GlobalPartitionMatmul: 'static + Send + Sync {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        partition_span: PartitionSpan,
        acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    );
}

#[derive(CubeType)]
/// Iterates on global matmuls in a row major fashion
pub struct RowMajorGlobalPartitionMatmul {}

#[derive(CubeType)]
/// Iterates on global matmuls in a col major fashion
pub struct ColMajorGlobalPartitionMatmul {}

#[derive(CubeType)]
/// Iterates on global matmuls following the swizzle algorithm
///
/// The swizzle algorithm processes  W elements per row in a top-down pass,
/// then shifts to the next W columns in a bottom-up pass.
/// This zigzag (top-down, bottom-up) repeats, covering the matrix span by span.
pub struct SwizzleGlobalPartitionMatmul<const W: u32> {}

#[cube]
impl PartitionSpan {
    pub fn new(
        row: PartitionSpanDim,
        col: PartitionSpanDim,
        batch: PartitionSpanDim,
    ) -> PartitionSpan {
        PartitionSpan { row, col, batch }
    }
}

#[cube]
impl PartitionSpanDim {
    pub fn new(
        problem_dim: u32,
        cube_pos: u32,
        #[comptime] stage_dim: u32,
        #[comptime] global_partition_dim: u32,
    ) -> PartitionSpanDim {
        let start = cube_pos * global_partition_dim;
        let end = Min::min(start + global_partition_dim, problem_dim);
        PartitionSpanDim {
            start,
            end,
            step: stage_dim,
            num_steps: global_partition_dim.div_ceil(stage_dim),
        }
    }
}

#[cube]
impl GlobalPartitionMatmul for RowMajorGlobalPartitionMatmul {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        span: PartitionSpan,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        #[unroll(span.batch.num_steps <= 1)]
        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            #[unroll(span.row.num_steps <= 1)]
            for row_iter in range_stepped(span.row.start, span.row.end, span.row.step) {
                #[unroll(span.col.num_steps <= 1)]
                for col_iter in range_stepped(span.col.start, span.col.end, span.col.step) {
                    GMM::zero_accumulator(&mut acc, config);
                    gmm_execute::<MP, GMM>(
                        lhs,
                        rhs,
                        out,
                        row_iter,
                        col_iter,
                        batch_iter,
                        &mut acc,
                        k_range,
                        quantization,
                        config,
                    );
                }
            }
        }
    }
}

#[cube]
impl GlobalPartitionMatmul for ColMajorGlobalPartitionMatmul {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        span: PartitionSpan,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        #[unroll(span.batch.num_steps <= 1)]
        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            #[unroll(span.col.num_steps <= 1)]
            for col_iter in range_stepped(span.col.start, span.col.end, span.col.step) {
                #[unroll(span.row.num_steps <= 1)]
                for row_iter in range_stepped(span.row.start, span.row.end, span.row.step) {
                    GMM::zero_accumulator(&mut acc, config);
                    gmm_execute::<MP, GMM>(
                        lhs,
                        rhs,
                        out,
                        row_iter,
                        col_iter,
                        batch_iter,
                        &mut acc,
                        k_range,
                        quantization,
                        config,
                    );
                }
            }
        }
    }
}

#[cube]
impl<const W: u32> GlobalPartitionMatmul for SwizzleGlobalPartitionMatmul<W> {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        span: PartitionSpan,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        let num_swizzle = comptime!(span.row.num_steps * span.col.num_steps);

        #[unroll(span.batch.num_steps <= 1)]
        for batch_iter in range_stepped(span.batch.start, span.batch.end, span.batch.step) {
            #[unroll(num_swizzle <= 1)]
            for n in 0..num_swizzle {
                GMM::zero_accumulator(&mut acc, config);
                let (row, col) = swizzle(n, span.row.num_steps, W);

                let row_iter = span.row.start + row * span.row.step;
                let col_iter = span.col.start + col * span.col.step;
                gmm_execute::<MP, GMM>(
                    lhs,
                    rhs,
                    out,
                    row_iter,
                    col_iter,
                    batch_iter,
                    &mut acc,
                    k_range,
                    quantization,
                    config,
                );
            }
        }
    }
}

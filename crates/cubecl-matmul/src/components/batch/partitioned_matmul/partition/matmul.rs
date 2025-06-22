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

#[derive(CubeType)]
/// Area of a tensor a cube is responsible of performing matmul
/// Similar to the concept of tensor slice, but specialized for matmul constraints
pub struct PartitionRanges {
    row: PartitionRangeDim,
    col: PartitionRangeDim,
    batch: PartitionRangeDim,
}

#[derive(CubeType)]
pub struct PartitionRangeDim {
    start: u32,
    #[cube(comptime)]
    step: u32,
    #[cube(comptime)]
    num_steps: u32,
}

#[cube]
/// Iterates on several global matmul across a partition
pub trait GlobalPartitionMatmul: 'static + Send + Sync {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        partition_ranges: PartitionRanges,
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
/// This zigzag (top-down, bottom-up) repeats, covering the matrix global matmul per global matmul.
pub struct SwizzleGlobalPartitionMatmul<const W: u32> {}

#[cube]
impl PartitionRanges {
    pub fn new(
        row: PartitionRangeDim,
        col: PartitionRangeDim,
        batch: PartitionRangeDim,
    ) -> PartitionRanges {
        PartitionRanges { row, col, batch }
    }
}

#[cube]
impl PartitionRangeDim {
    pub fn new(
        cube_pos: u32,
        #[comptime] stage_dim: u32,
        #[comptime] global_partition_dim: u32,
    ) -> PartitionRangeDim {
        let start = cube_pos * global_partition_dim;
        PartitionRangeDim {
            start,
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
        ranges: PartitionRanges,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        #[unroll(ranges.batch.num_steps <= 1)]
        for b in 0..ranges.batch.num_steps {
            let batch_iter = ranges.batch.start + b * ranges.batch.step;
            #[unroll(ranges.row.num_steps <= 1)]
            for r in 0..ranges.row.num_steps {
                let row_offset = ranges.row.start + r * ranges.row.step;
                #[unroll(ranges.col.num_steps <= 1)]
                for c in 0..ranges.col.num_steps {
                    let col_offset = ranges.col.start + c * ranges.col.step;

                    GMM::zero_accumulator(&mut acc, config);
                    gmm_execute::<MP, GMM>(
                        lhs,
                        rhs,
                        out,
                        row_offset,
                        col_offset,
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
        ranges: PartitionRanges,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        #[unroll(ranges.batch.num_steps <= 1)]
        for b in 0..ranges.batch.num_steps {
            let batch_iter = ranges.batch.start + b * ranges.batch.step;
            #[unroll(ranges.col.num_steps <= 1)]
            for c in 0..ranges.col.num_steps {
                let col_offset = ranges.col.start + c * ranges.col.step;
                #[unroll(ranges.row.num_steps <= 1)]
                for r in 0..ranges.row.num_steps {
                    let row_offset = ranges.row.start + r * ranges.row.step;

                    GMM::zero_accumulator(&mut acc, config);
                    gmm_execute::<MP, GMM>(
                        lhs,
                        rhs,
                        out,
                        row_offset,
                        col_offset,
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
        ranges: PartitionRanges,
        mut acc: GMM::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: GMM::Config,
    ) {
        let num_swizzle = comptime!(ranges.row.num_steps * ranges.col.num_steps);

        #[unroll(ranges.batch.num_steps <= 1)]
        for b in 0..ranges.batch.num_steps {
            let batch_iter = ranges.batch.start + b * ranges.batch.step;

            #[unroll(num_swizzle <= 1)]
            for n in 0..num_swizzle {
                GMM::zero_accumulator(&mut acc, config);
                let (row, col) = swizzle(n, ranges.row.num_steps, W);

                let row_offset = ranges.row.start + row * ranges.row.step;
                let col_offset = ranges.col.start + col * ranges.col.step;
                gmm_execute::<MP, GMM>(
                    lhs,
                    rhs,
                    out,
                    row_offset,
                    col_offset,
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

#[cube]
/// Execute global matmul on lhs, rhs, writing in out.
/// x and y offsets are absolute rows and columns
pub(crate) fn gmm_execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
    lhs: VirtualTensor<MP::EI>,
    rhs: VirtualTensor<MP::EI>,
    out: VirtualTensor<MP::EO, ReadWrite>,
    x_offset: u32,
    y_offset: u32,
    nth_batch: u32,
    acc: &mut GMM::Accumulator,
    k_range: (u32, u32),
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] config: GMM::Config,
) {
    let rank = out.rank();

    let batch_out = nth_batch * out.stride(rank - 2) * out.shape(rank - 2);
    let mut batch_lhs = 0u32.runtime();
    let mut batch_rhs = 0u32.runtime();
    for axis in 0..rank - 2 {
        let tmp = batch_out / out.stride(axis);
        batch_lhs += tmp % lhs.shape(axis) * lhs.stride(axis);
        batch_rhs += tmp % rhs.shape(axis) * rhs.stride(axis);
    }

    GMM::execute(
        GMM::init_lhs_loader(
            lhs,
            x_offset,
            k_range.0,
            nth_batch,
            batch_lhs,
            quantization,
            config,
        ),
        GMM::init_rhs_loader(
            rhs,
            k_range.0,
            y_offset,
            nth_batch,
            batch_rhs,
            quantization,
            config,
        ),
        GMM::init_writer(out, x_offset, y_offset, nth_batch, batch_out),
        acc,
        k_range,
        config,
    );
}

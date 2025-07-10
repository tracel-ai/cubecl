use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    MatmulPrecision,
    global::{self, Quantization},
};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

#[derive(CubeType)]
/// Area of a tensor a cube is responsible of performing matmul
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
/// Iterates on several global matmul across a global partition
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

#[cube]
impl PartitionRanges {
    /// Create a new [PartitionRanges]
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
    /// Create a new [PartitionRangeDim]
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
        // Needed for the unroll macro to work.
        let num_steps_batch = comptime!(ranges.batch.num_steps);
        let num_steps_row = comptime!(ranges.row.num_steps);
        let num_steps_col = comptime!(ranges.col.num_steps);

        #[unroll(num_steps_batch == 1)]
        for b in 0..num_steps_batch {
            let batch_iter = ranges.batch.start + b * ranges.batch.step;

            #[unroll(num_steps_row == 1)]
            for r in 0..num_steps_row {
                let row_offset = ranges.row.start + r * ranges.row.step;

                #[unroll(num_steps_col == 1)]
                for c in 0..num_steps_col {
                    let col_offset = ranges.col.start + c * ranges.col.step;

                    execute_global_matmul::<MP, GMM>(
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
        // Needed for the unroll macro to work.
        let num_steps_batch = comptime!(ranges.batch.num_steps);
        let num_steps_row = comptime!(ranges.row.num_steps);
        let num_steps_col = comptime!(ranges.col.num_steps);

        #[unroll(num_steps_batch == 1)]
        for b in 0..num_steps_batch {
            let batch_iter = ranges.batch.start + b * ranges.batch.step;

            #[unroll(num_steps_col == 1)]
            for c in 0..num_steps_col {
                let col_offset = ranges.col.start + c * ranges.col.step;

                #[unroll(num_steps_row == 1)]
                for r in 0..num_steps_row {
                    let row_offset = ranges.row.start + r * ranges.row.step;

                    execute_global_matmul::<MP, GMM>(
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
/// Execute global matmul on lhs, rhs, writing in out.
/// m and n offsets are absolute rows and columns
pub(crate) fn execute_global_matmul<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
    lhs: VirtualTensor<MP::EI>,
    rhs: VirtualTensor<MP::EI>,
    out: VirtualTensor<MP::EO, ReadWrite>,
    m_offset: u32,
    n_offset: u32,
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
            m_offset,
            k_range.0,
            nth_batch,
            batch_lhs,
            quantization,
            config,
        ),
        GMM::init_rhs_loader(
            rhs,
            k_range.0,
            n_offset,
            nth_batch,
            batch_rhs,
            quantization,
            config,
        ),
        GMM::init_writer(out, m_offset, n_offset, nth_batch, batch_out),
        acc,
        k_range,
        config,
    );
}

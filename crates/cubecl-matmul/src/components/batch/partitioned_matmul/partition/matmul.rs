use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    AccG, LhsG, MatmulPrecision, RhsG,
    global::{self, GlobalConfig},
};
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};

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
        a: VirtualTensor<LhsG<MP>>,
        b: VirtualTensor<RhsG<MP>>,
        c: CubeOption<VirtualTensor<AccG<MP>>>,
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        partition_ranges: PartitionRanges,
        acc: GMM::Accumulators,
        k_range: (u32, u32),
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
        a: VirtualTensor<LhsG<MP>>,
        b: VirtualTensor<RhsG<MP>>,
        c: CubeOption<VirtualTensor<AccG<MP>>>,
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        ranges: PartitionRanges,
        mut acc: GMM::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        // Needed for the unroll macro to work.
        let num_steps_batch = comptime!(ranges.batch.num_steps);
        let num_steps_row = comptime!(ranges.row.num_steps);
        let num_steps_col = comptime!(ranges.col.num_steps);

        #[unroll(num_steps_batch == 1)]
        for batch in 0..num_steps_batch {
            let batch_iter = ranges.batch.start + batch * ranges.batch.step;

            #[unroll(num_steps_row == 1)]
            for row in 0..num_steps_row {
                let row_offset = ranges.row.start + row * ranges.row.step;

                #[unroll(num_steps_col == 1)]
                for col in 0..num_steps_col {
                    let col_offset = ranges.col.start + col * ranges.col.step;

                    execute_global_matmul::<MP, GMM>(
                        a, b, c, out, row_offset, col_offset, batch_iter, &mut acc, k_range, config,
                    );
                }
            }
        }
    }
}

#[cube]
impl GlobalPartitionMatmul for ColMajorGlobalPartitionMatmul {
    fn execute<MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        a: VirtualTensor<LhsG<MP>>,
        b: VirtualTensor<RhsG<MP>>,
        c: CubeOption<VirtualTensor<AccG<MP>>>,
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        ranges: PartitionRanges,
        mut acc: GMM::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: GMM::Config,
    ) {
        // Needed for the unroll macro to work.
        let num_steps_batch = comptime!(ranges.batch.num_steps);
        let num_steps_row = comptime!(ranges.row.num_steps);
        let num_steps_col = comptime!(ranges.col.num_steps);

        #[unroll(num_steps_batch == 1)]
        for batch in 0..num_steps_batch {
            let batch_iter = ranges.batch.start + batch * ranges.batch.step;

            #[unroll(num_steps_col == 1)]
            for col in 0..num_steps_col {
                let col_offset = ranges.col.start + col * ranges.col.step;

                #[unroll(num_steps_row == 1)]
                for row in 0..num_steps_row {
                    let row_offset = ranges.row.start + row * ranges.row.step;

                    execute_global_matmul::<MP, GMM>(
                        a, b, c, out, row_offset, col_offset, batch_iter, &mut acc, k_range, config,
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
    a: VirtualTensor<LhsG<MP>>,
    b: VirtualTensor<RhsG<MP>>,
    c: CubeOption<VirtualTensor<AccG<MP>>>,
    out: VirtualTensor<AccG<MP>, ReadWrite>,
    m_offset: u32,
    n_offset: u32,
    nth_batch: u32,
    acc: &mut GMM::Accumulators,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    let rank = out.rank();

    let batch_out = nth_batch * out.stride(rank - 2) * out.shape(rank - 2);
    let mut batch_a = 0u32.runtime();
    let mut batch_b = 0u32.runtime();
    for axis in 0..rank - 2 {
        let tmp = batch_out / out.stride(axis);
        batch_a += tmp % a.shape(axis) * a.stride(axis);
        batch_b += tmp % b.shape(axis) * b.stride(axis);
    }

    let tiling = config.tiling_scheme();
    let stage_m = tiling.elements_in_stage_m().runtime();
    let stage_n = tiling.elements_in_stage_n().runtime();
    let k_size = k_range.1 - k_range.0;

    GMM::execute(
        GMM::init_lhs_global_reader(
            a,
            batch_a,
            (m_offset, k_range.0),
            (stage_m, k_size),
            nth_batch,
            config,
        ),
        GMM::init_rhs_global_reader(
            b,
            batch_b,
            (k_range.0, n_offset),
            (k_size, stage_n),
            nth_batch,
            config,
        ),
        GMM::init_acc_global_reader(
            c,
            batch_out,
            (m_offset, n_offset),
            (stage_m, stage_n),
            nth_batch,
            config,
        ),
        GMM::init_global_writer(
            out,
            batch_out,
            (m_offset, n_offset),
            (stage_m, stage_n),
            nth_batch,
            config,
        ),
        acc,
        k_range,
        config,
    );
}

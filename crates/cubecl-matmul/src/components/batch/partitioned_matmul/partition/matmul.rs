use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    AccG, LhsG, MatmulPrecision, RhsG,
    batch::SliceIndex,
    global::{self, GlobalConfig, args::BatchedMatrix},
};
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords3d},
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
        a: BatchedMatrix<LhsG<MP>>,
        b: BatchedMatrix<RhsG<MP>>,
        c: CubeOption<BatchedMatrix<AccG<MP>>>,
        out: View<Line<AccG<MP>>, Coords3d, ReadWrite>,
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
        a: BatchedMatrix<LhsG<MP>>,
        b: BatchedMatrix<RhsG<MP>>,
        c: CubeOption<BatchedMatrix<AccG<MP>>>,
        out: View<Line<AccG<MP>>, Coords3d, ReadWrite>,
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
                        a.cloned(),
                        b.cloned(),
                        c.cloned(),
                        out,
                        batch_iter,
                        row_offset,
                        col_offset,
                        &mut acc,
                        k_range,
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
        a: BatchedMatrix<LhsG<MP>>,
        b: BatchedMatrix<RhsG<MP>>,
        c: CubeOption<BatchedMatrix<AccG<MP>>>,
        out: View<Line<AccG<MP>>, Coords3d, ReadWrite>,
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
                        a.cloned(),
                        b.cloned(),
                        c.cloned(),
                        out,
                        batch_iter,
                        row_offset,
                        col_offset,
                        &mut acc,
                        k_range,
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
    a: BatchedMatrix<LhsG<MP>>,
    b: BatchedMatrix<RhsG<MP>>,
    c: CubeOption<BatchedMatrix<AccG<MP>>>,
    out: View<Line<AccG<MP>>, Coords3d, ReadWrite>,
    nth_batch: u32,
    m_offset: u32,
    n_offset: u32,
    acc: &mut GMM::Accumulators,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    let tiling = config.tiling_scheme();
    let stage_m = tiling.elements_in_stage_m().runtime();
    let stage_n = tiling.elements_in_stage_n().runtime();
    let k_size = k_range.1 - k_range.0;

    let a = a.into_matrix(nth_batch);
    let b = b.into_matrix(nth_batch);
    let c = match c {
        CubeOption::Some(c) => {
            let c = c.into_matrix(nth_batch);
            CubeOption::new_Some(c.slice_unchecked((m_offset, n_offset), (stage_m, stage_n)))
        }
        CubeOption::None => CubeOption::new_None(),
    };
    let out = out.view_mut(SliceIndex::new(nth_batch, out.shape()));

    GMM::execute(
        GMM::init_lhs_global_reader(
            a.slice_unchecked((m_offset, k_range.0), (stage_m, k_size)),
            config,
        ),
        GMM::init_rhs_global_reader(
            b.slice_unchecked((k_range.0, n_offset), (k_size, stage_n)),
            config,
        ),
        GMM::init_acc_global_reader(c, config),
        GMM::init_global_writer(
            out.slice_mut_unchecked((m_offset, n_offset), (stage_m, stage_n)),
            config,
        ),
        acc,
        k_range,
        config,
    );
}

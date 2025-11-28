use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{
    AccG, LhsG, MatmulPrecision, RhsG,
    batch::SliceIndex,
    global::{self, GlobalConfig, args::MatmulArgs},
    stage::StageConfig,
};
use cubecl_std::{CubeOption, CubeOptionExpand};

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
    fn execute<Args: MatmulArgs, MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        partition_ranges: PartitionRanges,
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
        #[comptime] global_partition_size: u32,
    ) -> PartitionRangeDim {
        PartitionRangeDim {
            start: cube_pos * global_partition_size * stage_dim,
            step: stage_dim,
            num_steps: global_partition_size,
        }
    }
}

#[cube]
impl GlobalPartitionMatmul for RowMajorGlobalPartitionMatmul {
    fn execute<Args: MatmulArgs, MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        ranges: PartitionRanges,
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

                    execute_global_matmul::<Args, MP, GMM>(
                        state, batch_iter, row_offset, col_offset, k_range, config,
                    );
                }
            }
        }
    }
}

#[cube]
impl GlobalPartitionMatmul for ColMajorGlobalPartitionMatmul {
    fn execute<Args: MatmulArgs, MP: MatmulPrecision, GMM: global::GlobalMatmul<MP>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        ranges: PartitionRanges,
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

                    execute_global_matmul::<Args, MP, GMM>(
                        state, batch_iter, row_offset, col_offset, k_range, config,
                    );
                }
            }
        }
    }
}

#[cube]
/// Execute global matmul on lhs, rhs, writing in out.
/// m and n offsets are absolute rows and columns
pub(crate) fn execute_global_matmul<
    Args: MatmulArgs,
    MP: MatmulPrecision,
    GMM: global::GlobalMatmul<MP>,
>(
    state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
    nth_batch: u32,
    m_offset: u32,
    n_offset: u32,
    k_range: (u32, u32),
    #[comptime] config: GMM::Config,
) {
    let stage_m = config.stage_config().elements_in_stage_m().runtime();
    let stage_n = config.stage_config().elements_in_stage_n().runtime();
    let k_size = k_range.1 - k_range.0;

    let a = Args::view_lhs(state);
    let b = Args::view_rhs(state);
    let c = Args::view_acc(state);
    let out = Args::view_out(state);

    let a_batch = Args::batch_lhs(state, nth_batch);
    let a = a.view(SliceIndex::new(a_batch, a.shape()));
    let b_batch = Args::batch_rhs(state, nth_batch);
    let b = b.view(SliceIndex::new(b_batch, b.shape()));
    let c_batch = Args::batch_acc(state, nth_batch);
    let c = match c {
        CubeOption::Some(c) => {
            let c = c.view(SliceIndex::new(c_batch, c.shape()));
            CubeOption::new_Some(c.slice_unchecked((m_offset, n_offset), (stage_m, stage_n)))
        }
        CubeOption::None => CubeOption::new_None(),
    };
    let out_batch = Args::batch_out(state, nth_batch);
    let out = out.view_mut(SliceIndex::new(out_batch, out.shape()));

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
        k_range,
        config,
    );
}

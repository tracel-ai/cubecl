use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::fragment::AttentionMatmulConfig;
use crate::components::fragment::FragmentLayout;
use crate::components::fragment::{FragmentLayoutExpand, FragmentOps, FragmentOpsExpand};
use crate::components::tile::ReduceOp;
use crate::components::tile::Reducer;
use crate::components::tile::RowWise;

#[derive(CubeType)]
/// Naive row reducer using shared memory
pub struct NaiveReducer {}

#[cube]
impl Reducer for NaiveReducer {
    fn reduce<E: Float, F: FragmentOps<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] config: TC,
    ) {
        let num_vals_in_plane = config.num_rows_per_unit() * config.plane_dim();
        let mut smem = SharedMemory::<E>::new(num_vals_in_plane * config.num_planes());

        let local_vals = RO::reduce_local::<F>(data);

        let plane_offset = UNIT_POS_Y * num_vals_in_plane;
        let unit_offset = UNIT_POS_X;

        #[unroll]
        for r in 0..config.num_rows_per_unit() {
            let row_offset = r * config.plane_dim();
            let offset = plane_offset + row_offset + unit_offset;

            smem[offset] = local_vals.index(r);
        }

        sync_cube();

        let num_units_per_row = data.layout().num_units_per_row();

        #[unroll]
        for r in 0..config.num_rows_per_unit() {
            let mut val = vals.index(r);

            let row_offset = r * config.plane_dim();

            for c in 0..num_units_per_row {
                let unit_offset = (UNIT_POS_X / num_units_per_row) * num_units_per_row;
                let offset = plane_offset + row_offset + unit_offset;

                val = RO::reduce_step_scalar(val, smem[offset + c]);
            }

            vals.replace_at(r, val);
        }

        sync_cube();
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::ReduceOp;
use crate::components::tile::Reducer;
use crate::components::tile::RowWise;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

#[derive(CubeType)]
pub struct DummyReducer {}

#[cube]
impl Reducer for DummyReducer {
    fn reduce<E: Float, PL: PlaneLayout<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &PL,
        #[comptime] config: TC,
    ) {
        let num_vals_in_plane = config.num_rows_per_unit() * config.plane_dim();
        let mut smem = SharedMemory::<E>::new(num_vals_in_plane * config.num_planes());

        let local_vals = RO::reduce_local::<PL>(data);

        let plane_offset = UNIT_POS_Y * num_vals_in_plane;
        let unit_offset = UNIT_POS_X;

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let row_offset = r * config.plane_dim();
            let offset = plane_offset + row_offset + unit_offset;

            smem[offset] = local_vals.index(r);

            comptime![r += 1];
        }

        sync_cube();

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let mut val = vals.index(r);

            let row_offset = r * config.plane_dim();

            for c in 0..data.num_units_per_row() {
                let unit_offset =
                    (UNIT_POS_X / data.num_units_per_row()) * data.num_units_per_row();
                let offset = plane_offset + row_offset + unit_offset;

                val = RO::reduce_step_scalar(val, smem[offset + c]);
            }

            vals.replace_at(r, val);

            comptime![r += 1];
        }

        sync_cube();
    }
}

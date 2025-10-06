use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::PlaneOps;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowVal, RowWise};

#[derive(CubeType)]
pub struct DummyReducer {}

#[cube]
impl PlaneOps for DummyReducer {
    fn row_sum<E: Float, PL: PlaneLayout<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &PL,
        #[comptime] config: TC,
    ) {
        let num_sums_in_plane = config.num_rows_per_unit() * config.plane_dim();
        let mut smem = SharedMemory::<E>::new(num_sums_in_plane * config.num_planes());

        let local_sums = data.rowwise_sum();

        let plane_offset = UNIT_POS_Y * num_sums_in_plane;
        let unit_offset = UNIT_POS_X;

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let row_offset = r * config.plane_dim();
            let offset = plane_offset + row_offset + unit_offset;

            smem[offset] = local_sums.index(r);

            comptime![r += 1];
        }

        sync_cube();

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let mut val = E::from_int(0);
            let row_offset = r * config.plane_dim();

            for c in 0..data.num_units_per_row() {
                let unit_offset =
                    (UNIT_POS_X / data.num_units_per_row()) * data.num_units_per_row();
                let offset = plane_offset + row_offset + unit_offset;

                val += smem[offset + c];
            }

            vals.replace_at(r, val);

            comptime![r += 1];
        }
    }

    fn row_max<E: Float, PL: PlaneLayout<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        base: &RowWise<E>,
        data: &PL,
        #[comptime] config: TC,
    ) {
        let num_maxes_in_plane = config.num_rows_per_unit() * config.plane_dim();
        let mut smem = SharedMemory::<E>::new(num_maxes_in_plane * config.num_planes());

        let local_maxes = data.rowwise_max();

        let plane_offset = UNIT_POS_Y * num_maxes_in_plane;
        let unit_offset = UNIT_POS_X;

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let row_offset = r * config.plane_dim();
            let offset = plane_offset + row_offset + unit_offset;

            smem[offset] = local_maxes.index(r);

            comptime![r += 1];
        }

        sync_cube();

        let mut r = comptime![0u32];

        #[unroll]
        for _ in 0..config.num_rows_per_unit() {
            let mut val = base.index(r);
            let row_offset = r * config.plane_dim();

            for c in 0..data.num_units_per_row() {
                let unit_offset =
                    (UNIT_POS_X / data.num_units_per_row()) * data.num_units_per_row();
                let offset = plane_offset + row_offset + unit_offset;

                val = Max::max(val, smem[offset + c]);
            }

            vals.replace_at(r, val);

            comptime![r += 1];
        }
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::ReduceOp;
use crate::components::tile::Reducer;
use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowVal, RowWise};

#[derive(CubeType)]
pub struct BroadcastReducer {}

#[cube]
impl Reducer for BroadcastReducer {
    fn reduce<E: Float, PL: PlaneLayout<E>, RO: ReduceOp<E>, TC: AttentionMatmulConfig>(
        vals: &mut RowWise<E>,
        data: &PL,
        #[comptime] config: TC,
    ) {
        let num_units_per_row = data.num_units_per_row();
        let num_shares_within_plane = comptime!((num_units_per_row as f32).log2().ceil() as u32);

        let unit_pos = UNIT_POS_X;
        let unit_pos_in_row = unit_pos % num_units_per_row;

        let mut fpb = FakePlaneBroadcast::<E>::new(config.plane_dim(), config.num_planes());

        RO::reduce_local_store::<PL>(data, vals);

        for i in 0..num_shares_within_plane {
            let offset = num_units_per_row >> (i + 1);
            let source_unit = unit_pos + offset;

            let value_from_source = fpb.plane_broadcast(vals, source_unit);

            // Mask if outside the row
            let mask = unit_pos_in_row + offset >= num_units_per_row;
            RO::reduce_step_rowwise(vals, &value_from_source, mask);
        }

        // Broadcast back to subgroup
        let result = &fpb.plane_broadcast(vals, unit_pos - unit_pos_in_row);
        vals.copy_from(result);
    }
}

#[derive(CubeType)]
struct FakePlaneBroadcast<E: Float> {
    slice: SliceMut<E>,
}

#[cube]
impl<E: Float> FakePlaneBroadcast<E> {
    pub fn new(#[comptime] plane_dim: u32, #[comptime] num_planes: u32) -> Self {
        let mut smem = SharedMemory::<E>::new(plane_dim * num_planes);
        let start = UNIT_POS_Y * plane_dim;
        let end = start + plane_dim;
        FakePlaneBroadcast::<E> {
            slice: smem.slice_mut(start, end),
        }
    }

    pub fn plane_broadcast(&mut self, val: &RowWise<E>, source_unit: u32) -> RowWise<E> {
        let mut result = Sequence::new();

        let mut row = comptime![0];

        #[unroll]
        for _ in 0..val.num_rows {
            self.slice[UNIT_POS_X] = val.index(row);
            sync_cube();

            result.push(RowVal::<E> {
                val: self.slice[source_unit],
            });
            sync_cube();

            comptime![row += 1];
        }

        RowWise::<E> {
            num_rows: val.num_rows,
            vals: result,
        }
    }
}

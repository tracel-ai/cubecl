use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::stage::{ReduceOp, Reducer};
use crate::components::tile::{FragmentAttentionConfig, RowVal, RowWise};
use crate::components::tile::{RowwiseFormat, RowwiseFormatExpand};

#[derive(CubeType)]
/// Applies reduction on rows, masking planes that do not participate in the row
///
/// TODO: uses shared memory to plane_broadcast, should be replaced with
/// a plane primitive
pub struct BroadcastReducer {}

#[cube]
impl Reducer for BroadcastReducer {
    fn reduce<E: Float, F: RowwiseFormat<E>, RO: ReduceOp<E>, FC: FragmentAttentionConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] config: FC,
    ) {
        let num_units_per_row = data.num_units_per_row();
        let num_shares_within_plane = comptime!((num_units_per_row as f32).log2().ceil() as u32);

        let unit_pos = UNIT_POS_X;
        let unit_pos_in_row = unit_pos % num_units_per_row;

        let mut fpb = PlaneBroadcast::<E>::new(config.plane_dim(), config.num_planes());

        RO::reduce_local_accumulate::<F>(data, vals);

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
enum PlaneBroadcast<E: Float> {
    Genuine(GenuinePlaneBroadcast<E>),
    Fake(FakePlaneBroadcast<E>),
}

#[cube]
impl<E: Float> PlaneBroadcast<E> {
    pub fn new(#[comptime] plane_dim: u32, #[comptime] num_planes: u32) -> Self {
        // PlaneBroadcast::new_Fake(FakePlaneBroadcast::new(plane_dim, num_planes))
        PlaneBroadcast::new_Genuine(GenuinePlaneBroadcast::new())
    }

    pub fn plane_broadcast(&mut self, val: &RowWise<E>, source_unit: u32) -> RowWise<E> {
        match self {
            PlaneBroadcast::Genuine(pb) => pb.plane_broadcast(val, source_unit),
            PlaneBroadcast::Fake(pb) => pb.plane_broadcast(val, source_unit),
        }
    }
}

#[derive(CubeType)]
struct GenuinePlaneBroadcast<E: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<E>,
}

#[cube]
impl<E: Float> GenuinePlaneBroadcast<E> {
    pub fn new() -> Self {
        GenuinePlaneBroadcast::<E> {
            _phantom: PhantomData,
        }
    }

    pub fn plane_broadcast(&mut self, val: &RowWise<E>, source_unit: u32) -> RowWise<E> {
        let mut result = Sequence::new();

        #[unroll]
        for row in 0..val.num_rows {
            result.push(RowVal::<E> {
                val: E::cast_from(9u32), // val: plane_broadcast(val.index(row), source_unit),
            });
        }

        RowWise::<E> {
            num_rows: val.num_rows,
            vals: result,
        }
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

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};

#[derive(CubeType)]
pub struct RowVals<E: Float> {
    #[cube(comptime)]
    num_rows: u32,
    vals: Sequence<RowVal<E>>,
}

#[derive(CubeType)]
pub struct RowVal<E: Float> {
    val: E,
}

#[cube]
impl<E: Float> RowWise for RowVals<E> {
    type E = E;

    fn new_filled(#[comptime] num_rows: u32, val: E) -> RowVals<E> {
        let mut vals = Sequence::new();
        #[unroll]
        for _ in 0..num_rows {
            vals.push(RowVal::<E> { val });
        }
        RowVals::<E> { num_rows, vals }
    }

    fn new_min_value(#[comptime] num_rows: u32) -> RowVals<E> {
        Self::new_filled(num_rows, E::min_value())
    }

    fn new_zero(#[comptime] num_rows: u32) -> RowVals<E> {
        Self::new_filled(num_rows, E::from_int(0))
    }

    fn copy_from(this: &mut Self, other: &RowVals<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..this.num_rows {
            let row_val = this.vals.index_mut(i);
            row_val.val = other.index(i);

            comptime![i += 1];
        }
    }

    fn index(&self, i: u32) -> Self::E {
        self.vals.index(i).val
    }

    fn fill(this: &mut Self, val: Self::E) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..this.num_rows {
            let row_val = this.vals.index_mut(i);
            row_val.val = val;

            comptime![i += 1];
        }
    }

    fn row_sum<PL: PlaneLayout<E = Self::E>, TC: AttentionMatmulConfig>(
        placeholder: &mut Self,
        data: &PL,
        #[comptime] config: TC,
    ) {
        Self::fill(placeholder, <RowSum as RowOp<PL>>::neutral_element());
        row_op::<PL, RowSum, TC>(&mut placeholder.vals, data, config)
    }

    fn row_max<PL: PlaneLayout<E = Self::E>, TC: AttentionMatmulConfig>(
        placeholder: &mut Self,
        base: &Self,
        data: &PL,
        #[comptime] config: TC,
    ) {
        Self::copy_from(placeholder, base);
        row_op::<PL, RowMax, TC>(&mut placeholder.vals, data, config)
    }
}

#[cube]
trait RowOp<PL: PlaneLayout> {
    fn neutral_element() -> PL::E;

    fn update(acc: PL::E, val: PL::E, mask: bool) -> PL::E;
}

#[derive(CubeType)]
struct RowMax {}

#[derive(CubeType)]
struct RowSum {}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowMax {
    fn neutral_element() -> PL::E {
        PL::E::min_value()
    }

    fn update(acc: PL::E, val: PL::E, mask: bool) -> PL::E {
        Max::max(acc, val + PL::E::cast_from(mask) * PL::E::min_value())
    }
}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowSum {
    fn neutral_element() -> PL::E {
        PL::E::from_int(0)
    }

    fn update(acc: PL::E, val: PL::E, mask: bool) -> PL::E {
        acc + val * PL::E::cast_from(!mask)
    }
}

#[cube]
fn row_op<PL: PlaneLayout, RO: RowOp<PL>, TC: AttentionMatmulConfig>(
    vals: &mut Sequence<RowVal<PL::E>>,
    data: &PL,
    #[comptime] config: TC,
) {
    let num_local_rows = data.num_local_rows();
    let num_local_cols = data.num_local_cols();
    let num_units_per_row = data.num_units_per_row();
    let num_shares_within_plane = comptime!((num_units_per_row as f32).log2().ceil() as u32);

    let unit_pos = UNIT_POS_X;
    let unit_pos_in_row = unit_pos % num_units_per_row;

    let mut fpb = FakePlaneBroadcast::<PL::E>::new(config.plane_dim(), config.num_planes());

    let mut local_row = comptime![0];

    #[unroll]
    for _ in 0..num_local_rows {
        // let mut local_val = RO::neutral_element();
        let mut local_val = vals.index(0u32).val;

        #[unroll]
        for local_col in 0..num_local_cols {
            local_val = RO::update(local_val, data.get_at_coor(local_row, local_col), false);
        }

        let mut total_val = local_val;

        for i in 0..num_shares_within_plane {
            let offset = num_units_per_row >> (i + 1);
            let source_unit = unit_pos + offset;

            let value_from_source = fpb.plane_broadcast(total_val, source_unit);

            // Mask if outside the row
            let mask = unit_pos_in_row + offset >= num_units_per_row;
            total_val = RO::update(total_val, value_from_source, mask);
        }

        // Broadcast back to subgroup
        let row_result = fpb.plane_broadcast(total_val, unit_pos - unit_pos_in_row);
        let row_val = vals.index_mut(local_row);
        row_val.val = row_result;

        comptime![local_row += 1];
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

    pub fn plane_broadcast(&mut self, val: E, source_unit: u32) -> E {
        self.slice[UNIT_POS_X] = val;
        sync_cube();

        let result = self.slice[source_unit];
        sync_cube();

        result
    }
}

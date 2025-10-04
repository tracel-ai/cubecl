use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::dummy::AttentionMatmulConfig;
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

#[derive(CubeType)]
pub struct RowWise<E: Float> {
    #[cube(comptime)]
    num_rows: u32,
    vals: Sequence<RowVal<E>>,
}

#[derive(CubeType)]
pub struct RowVal<E: Float> {
    val: E,
}

#[cube]
impl<E: Float> RowWise<E> {
    pub fn new_filled(#[comptime] num_rows: u32, val: E) -> RowWise<E> {
        let mut vals = Sequence::new();
        #[unroll]
        for _ in 0..num_rows {
            vals.push(RowVal::<E> { val });
        }
        RowWise::<E> { num_rows, vals }
    }

    pub fn new_min_value(#[comptime] num_rows: u32) -> RowWise<E> {
        Self::new_filled(num_rows, E::min_value())
    }

    pub fn new_zero(#[comptime] num_rows: u32) -> RowWise<E> {
        Self::new_filled(num_rows, E::from_int(0))
    }

    pub fn copy_from(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = other.index(i);

            comptime![i += 1];
        }
    }

    pub fn index(&self, i: u32) -> E {
        self.vals.index(i).val
    }

    pub fn fill(&mut self, val: E) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = val;

            comptime![i += 1];
        }
    }

    pub fn add(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val += other.index(i);

            comptime![i += 1];
        }
    }

    pub fn mul(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val *= other.index(i);

            comptime![i += 1];
        }
    }

    pub fn max(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            Max::max(row_val.val, other.index(i));

            comptime![i += 1];
        }
    }
}

#[cube]
trait RowOp<E: Float> {
    // fn neutral_element() -> RowWise<E>;

    fn reduce_local<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>);

    fn reduce_one(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}

#[derive(CubeType)]
struct RowMax {}

#[derive(CubeType)]
struct RowSum {}

#[cube]
impl<E: Float> RowOp<E> for RowMax {
    fn reduce_local<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>) {
        acc.max(&data.rowwise_max())
    }

    fn reduce_one(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(mask) * E::min_value());
        masked.add(&elem);

        acc.max(&masked)
    }
}

#[cube]
impl<E: Float> RowOp<E> for RowSum {
    fn reduce_local<PL: PlaneLayout<E>>(data: &PL, acc: &mut RowWise<E>) {
        acc.add(&data.rowwise_sum())
    }

    fn reduce_one(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(!mask));
        masked.mul(&elem);

        acc.add(&masked)
    }
}

#[cube]
fn row_op<E: Float, PL: PlaneLayout<E>, RO: RowOp<E>, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    let num_units_per_row = data.num_units_per_row();
    let num_shares_within_plane = comptime!((num_units_per_row as f32).log2().ceil() as u32);

    let unit_pos = UNIT_POS_X;
    let unit_pos_in_row = unit_pos % num_units_per_row;

    let mut fpb = FakePlaneBroadcast::<E>::new(config.plane_dim(), config.num_planes());

    RO::reduce_local::<PL>(data, vals);

    for i in 0..num_shares_within_plane {
        let offset = num_units_per_row >> (i + 1);
        let source_unit = unit_pos + offset;

        let value_from_source = fpb.plane_broadcast(&vals, source_unit);

        // Mask if outside the row
        let mask = unit_pos_in_row + offset >= num_units_per_row;
        RO::reduce_one(vals, &value_from_source, mask);
    }

    // Broadcast back to subgroup
    let result = &fpb.plane_broadcast(&vals, unit_pos - unit_pos_in_row);
    vals.copy_from(result);
}

#[cube]
pub fn row_sum<E: Float, PL: PlaneLayout<E>, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(&RowWise::new_zero(vals.num_rows));
    row_op::<E, PL, RowSum, TC>(vals, data, config)
}

#[cube]
pub fn row_max<E: Float, PL: PlaneLayout<E>, TC: AttentionMatmulConfig>(
    vals: &mut RowWise<E>,
    base: &RowWise<E>,
    data: &PL,
    #[comptime] config: TC,
) {
    vals.copy_from(base);
    row_op::<E, PL, RowMax, TC>(vals, data, config)
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
        for row in 0..val.num_rows {
            self.slice[UNIT_POS_X] = val.index(row);
            sync_cube();

            result.push(RowVal::<E> {
                val: self.slice[source_unit],
            });
            sync_cube();
        }
        RowWise::<E> {
            num_rows: val.num_rows,
            vals: result,
        }
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};
use crate::components::tile::{RowWise, RowWiseExpand};

#[derive(CubeType)]
/// Contains as many elements as there are rows in a tile.
/// Indices of rows where the unit does not participate are masked but take up space.
pub struct SparseArray<E: Float> {
    #[cube(comptime)]
    num_rows: u32,
    vals: Array<E>,
}

#[cube]
impl<E: Float> RowWise for SparseArray<E> {
    type E = E;

    fn new_filled(#[comptime] num_rows: u32, val: E) -> SparseArray<E> {
        let mut sparse_arr = SparseArray::<E> {
            num_rows,
            vals: Array::new(num_rows),
        };
        sparse_arr.fill(val);
        sparse_arr
    }

    fn new_min_value(#[comptime] num_rows: u32) -> SparseArray<E> {
        Self::new_filled(num_rows, E::min_value())
    }

    fn new_zero(#[comptime] num_rows: u32) -> SparseArray<E> {
        Self::new_filled(num_rows, E::from_int(0))
    }

    fn copy_from(this: &mut Self, other: &Self) {
        #[unroll]
        for i in 0..this.num_rows {
            this.vals[i] = other.vals[i];
        }
    }

    fn index(&self, i: u32) -> Self::E {
        self.vals[i]
    }

    fn row_sum<PL: PlaneLayout<E = Self::E>>(placeholder: &mut Self, data: &PL) {
        row_op::<PL, RowSum>(&mut placeholder.vals, data)
    }

    fn row_max<PL: PlaneLayout<E = Self::E>>(placeholder: &mut Self, base: &Self, data: &PL) {
        Self::copy_from(placeholder, base);
        row_op::<PL, RowMax>(&mut placeholder.vals, data)
    }
}

#[cube]
impl<E: Float> SparseArray<E> {
    pub fn single(val: E) -> SparseArray<E> {
        Self::new_filled(1u32, val)
    }

    pub fn fill(&mut self, val: E) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = val;
        }
    }

    pub fn cast_from<E2: Float>(&mut self, other: &SparseArray<E2>) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = E::cast_from(other.vals[i]);
        }
    }

    pub fn max_from(&mut self, other: &SparseArray<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = Max::max(self.vals[i], E::cast_from(other.vals[i]));
        }
    }

    pub fn vals_mut(&mut self) -> &mut Array<E> {
        &mut self.vals
    }
}

#[cube]
trait RowOp<PL: PlaneLayout> {
    fn mask(is_active: bool) -> PL::E;

    fn neutral_element() -> PL::E;

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E;

    fn plane_reduce(acc: PL::E) -> PL::E;
}

#[derive(CubeType)]
struct RowMax {}

#[derive(CubeType)]
struct RowSum {}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowMax {
    fn mask(is_active: bool) -> PL::E {
        PL::E::cast_from(!is_active) * PL::E::min_value()
    }

    fn neutral_element() -> PL::E {
        PL::E::min_value()
    }

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E {
        Max::max(acc, data.get_at_coor(row, col) + mask)
    }

    fn plane_reduce(acc: PL::E) -> PL::E {
        plane_max::<PL::E>(acc)
    }
}

#[cube]
impl<PL: PlaneLayout> RowOp<PL> for RowSum {
    fn mask(is_active: bool) -> PL::E {
        PL::E::cast_from(is_active)
    }

    fn neutral_element() -> PL::E {
        PL::E::from_int(0)
    }

    fn local_update(acc: PL::E, row: u32, col: u32, data: &PL, mask: PL::E) -> PL::E {
        // TODO BIG PROBLEM: get_at_coor is given absolute row, col but is implemented as if received local
        acc + data.get_at_coor(row, col) * mask
    }

    fn plane_reduce(acc: PL::E) -> PL::E {
        plane_sum::<PL::E>(acc)
    }
}

#[cube]
fn row_op<PL: PlaneLayout, RO: RowOp<PL>>(vals: &mut Array<PL::E>, data: &PL) {
    let total_row_count = data.total_rows_count();

    #[unroll]
    for row in 0..total_row_count {
        let is_active = data.is_owned(row);

        let mask = RO::mask(is_active);

        let mut local = RO::neutral_element();

        #[unroll]
        for c in 0..data.num_cols_per_unit() {
            let col = data.abs_col_index(row, c);
            local = RO::local_update(local, row, col, &data, mask);
        }

        vals[row] = RO::plane_reduce(local);
    }
}

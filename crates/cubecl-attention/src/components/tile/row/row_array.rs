use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::PlaneLayout;
use crate::components::tile::RowMax;
use crate::components::tile::RowSum;
use crate::components::tile::row_op;
use crate::components::tile::{RowWise, RowWiseExpand};

#[derive(CubeType)]
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

    fn copy_from(this: &mut Self, other: &Self) {
        #[unroll]
        for i in 0..this.num_rows {
            this.vals[i] = other.vals[i];
        }
    }

    fn index(&self, #[comptime] i: u32) -> Self::E {
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

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct RowWise<E: Float> {
    #[cube(comptime)]
    pub num_rows: u32,
    pub vals: Array<E>,
}

#[cube]
impl<E: Float> RowWise<E> {
    // pub fn new(#[comptime] num_rows: u32, vals: Array<E>) -> RowWise<E> {}

    pub fn new_filled(#[comptime] num_rows: u32, val: E) -> RowWise<E> {
        let mut row_wise = RowWise::<E> {
            num_rows,
            vals: Array::new(num_rows),
        };
        row_wise.fill(val);
        row_wise
    }

    pub fn single(val: E) -> RowWise<E> {
        Self::new_filled(1u32, val)
    }

    pub fn index(&self, #[comptime] i: u32) -> E {
        self.vals[i]
    }

    pub fn fill(&mut self, val: E) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = val;
        }
    }

    pub fn copy_from(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = other.vals[i];
        }
    }

    pub fn cast_from<E2: Float>(&mut self, other: &RowWise<E2>) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = E::cast_from(other.vals[i]);
        }
    }

    pub fn max_from(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            self.vals[i] = Max::max(self.vals[i], E::cast_from(other.vals[i]));
        }
    }
}

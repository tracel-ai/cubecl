use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct RowWise<E: Float> {
    #[cube(comptime)]
    pub num_rows: u32,
    pub vals: Sequence<RowVal<E>>,
}

#[derive(CubeType)]
pub struct RowVal<E: Float> {
    pub val: E,
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

    pub fn add_inplace(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val += other.index(i);

            comptime![i += 1];
        }
    }

    pub fn mul_inplace(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val *= other.index(i);

            comptime![i += 1];
        }
    }

    pub fn recip_inplace(&mut self) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = Recip::recip(row_val.val);

            comptime![i += 1];
        }
    }

    pub fn max_inplace(&mut self, other: &RowWise<E>) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = Max::max(row_val.val, other.index(i));

            comptime![i += 1];
        }
    }

    pub fn replace_at(&mut self, #[comptime] i: u32, new_val: E) {
        let row_val = self.vals.index_mut(i);
        row_val.val = new_val;
    }

    pub fn cast_from<E2: Float>(&self) -> RowWise<E2> {
        let mut vals = Sequence::new();
        let mut i = comptime![0u32];

        #[unroll]
        for _ in 0..self.num_rows {
            let val = E2::cast_from(self.index(i));
            vals.push(RowVal::<E2> { val });

            comptime![i += 1];
        }

        RowWise::<E2> {
            num_rows: self.num_rows,
            vals,
        }
    }

    pub fn exp_m_diff(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Sequence::new();
        let mut i = comptime![0u32];

        #[unroll]
        for _ in 0..self.num_rows {
            let val = Exp::exp(self.index(i) - other.index(i));
            vals.push(RowVal::<E> { val });

            comptime![i += 1];
        }

        RowWise::<E> {
            num_rows: self.num_rows,
            vals,
        }
    }

    pub fn mul(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Sequence::new();
        let mut i = comptime![0u32];

        #[unroll]
        for _ in 0..self.num_rows {
            let val = self.index(i) * other.index(i);
            vals.push(RowVal::<E> { val });

            comptime![i += 1];
        }

        RowWise::<E> {
            num_rows: self.num_rows,
            vals,
        }
    }

    pub fn add(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Sequence::new();
        let mut i = comptime![0u32];

        #[unroll]
        for _ in 0..self.num_rows {
            let val = self.index(i) + other.index(i);
            vals.push(RowVal::<E> { val });

            comptime![i += 1];
        }

        RowWise::<E> {
            num_rows: self.num_rows,
            vals,
        }
    }
}

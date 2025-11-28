use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::FULLY_MASKED_ROW_THRESHOLD;

#[derive(CubeType)]
/// Contains one value per row of a fragment for which the unit contributes
///
/// Example: For a 8x8 tile shared by a plane of 32 units,
/// every unit holds 8 values in the tile.
///
/// In the following layout, values are held contiguously, and num_rows=1 because
/// every two occurrences of the same plane id are in the same row
///  0,  0,  1,  1,  2,  2,  3,  3,
///  4,  4,  5,  5,  6,  6,  7,  7,
///  8,  8,  9,  9, 10, 10, 11, 11,
/// 12, 12, 13, 13, 14, 14, 15, 15,
/// 16, 16, 17, 17, 18, 18, 19, 19,
/// 20, 20, 21, 21, 22, 22, 23, 23,
/// 24, 24, 25, 25, 26, 26, 27, 27,
/// 28, 28, 29, 29, 30, 30, 31, 31,
///
/// In the following layout, values are held disjointly, and num_rows=2 because
/// the two occurrences of the same plane id are not in the same row
///  0,  1,  2,  3,  4,  5,  6,  7,
///  8,  9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23,
/// 24, 25, 26, 27, 28, 29, 30, 31,
///  0,  1,  2,  3,  4,  5,  6,  7,
///  8,  9, 10, 11, 12, 13, 14, 15,
/// 16, 17, 18, 19, 20, 21, 22, 23,
/// 24, 25, 26, 27, 28, 29, 30, 31,
pub struct RowWise<E: Numeric> {
    #[cube(comptime)]
    pub num_rows: u32,
    pub vals: Sequence<RowVal<E>>,
}

#[derive(CubeType)]
/// Wrapper over a value to enable mutating it
pub struct RowVal<E: Numeric> {
    pub val: E,
}

#[cube]
impl<E: Numeric> RowWise<E> {
    /// Create a RowWise with the provided value at every row
    pub fn new_filled(#[comptime] num_rows: u32, val: E) -> RowWise<E> {
        let mut vals = Sequence::new();
        #[unroll]
        for _ in 0..num_rows {
            vals.push(RowVal::<E> { val });
        }
        RowWise::<E> { num_rows, vals }
    }

    /// Fill the existing RowWise with the provided value at every row
    pub fn fill(&mut self, val: E) {
        #[unroll]
        for i in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = val;
        }
    }

    /// Create a RowWise with -infinity at every row
    pub fn new_min_value(#[comptime] num_rows: u32) -> RowWise<E> {
        Self::new_filled(num_rows, E::min_value())
    }

    /// Create a RowWise with zero at every row
    pub fn new_zero(#[comptime] num_rows: u32) -> RowWise<E> {
        Self::new_filled(num_rows, E::from_int(0))
    }

    /// Fill the current RowWise with the value of other at each row
    pub fn copy_from(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = other.index(i);
        }
    }

    /// Return the value at row i
    pub fn index(&self, i: u32) -> E {
        self.vals.index(i).val
    }

    /// For each row, add the the current and other, and outputs a new RowWise
    pub fn add(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for i in 0..self.num_rows {
            let val = self.index(i) + other.index(i);
            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.num_rows,
            vals,
        }
    }

    /// For each row, add the other value to the current RowWise
    pub fn add_inplace(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val += other.index(i);
        }
    }

    /// For each row, multiplies the the current and other, and outputs a new RowWise
    pub fn mul(&self, other: &RowWise<E>) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for i in 0..self.num_rows {
            let val = self.index(i) * other.index(i);
            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.num_rows,
            vals,
        }
    }

    /// For each row, multiplies the other value to the current RowWise
    pub fn mul_inplace(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val *= other.index(i);
        }
    }

    /// For each row, maxes the other value to the current RowWise
    pub fn max_inplace(&mut self, other: &RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);
            row_val.val = Max::max(row_val.val, other.index(i));
        }
    }

    /// Changes the value at index i
    pub fn replace_at(&mut self, #[comptime] i: u32, new_val: E) {
        let row_val = self.vals.index_mut(i);
        row_val.val = new_val;
    }

    /// Return a copy of self, cast into E2
    pub fn cast_from<E2: Float>(row_wise: &RowWise<E>) -> RowWise<E2> {
        let mut vals = Sequence::new();

        #[unroll]
        for i in 0..row_wise.num_rows {
            let val = E2::cast_from(row_wise.index(i));
            vals.push(RowVal::<E2> { val });
        }

        RowWise::<E2> {
            num_rows: row_wise.num_rows,
            vals,
        }
    }
}

#[cube]
impl<E: Float> RowWise<E> {
    /// Computes e^(self.val - other.val) for every row, and outputs a new RowWise
    pub fn exp_diff(&self, other: &RowWise<E>) -> RowWise<E> {
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

    /// Replaces each value `v` (v >= 0) in a row with `1/v`.
    ///
    /// If `v = 0`, the result is set to `0` instead of `1/0`.
    /// This occurs when the entire row is masked, meaning it should
    /// contribute no information, and ensures numerical stability.
    pub fn recip_inplace(&mut self) {
        let mut i = comptime![0u32];
        #[unroll]
        for _ in 0..self.num_rows {
            let row_val = self.vals.index_mut(i);

            let epsilon = E::new(FULLY_MASKED_ROW_THRESHOLD);
            let not_masked = E::cast_from(row_val.val >= epsilon);
            let safe_val = Max::max(row_val.val, epsilon);
            let recip = Recip::recip(safe_val);
            row_val.val = not_masked * recip;

            comptime![i += 1];
        }
    }
}

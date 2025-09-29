use crate::components::tile::RowElement;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct RowWise<E: Float> {
    #[cube(comptime)]
    num_rows: u32,
    vals: Sequence<RowVal<E>>,
}

#[cube]
impl<E: Float> RowElement<E> for RowWise<E> {
    fn copy(from: &Self, to: &mut Self) {
        from.copy_into(to);
    }
}

#[derive(CubeType, Copy, Clone)]
pub struct RowVal<E: Float> {
    val: E,
}

#[cube]
impl<E: Float> RowVal<E> {
    pub fn new(val: E) -> RowVal<E> {
        RowVal::<E> { val }
    }

    pub fn cast<E2: Float>(&self) -> RowVal<E2> {
        RowVal::<E2> {
            val: E2::cast_from(self.val),
        }
    }
}

#[cube]
impl<E: Float> RowWise<E> {
    pub fn new(#[comptime] num_rows: u32, vals: Sequence<RowVal<E>>) -> RowWise<E> {
        RowWise::<E> { num_rows, vals }
    }

    pub fn new_filled(#[comptime] num_rows: u32, val: E) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for _ in 0..num_rows {
            vals.push(RowVal::new(val));
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn single(val: E) -> RowWise<E> {
        let mut vals = Sequence::new();
        vals.push(RowVal::<E> { val });
        RowWise::<E>::new(1u32, vals)
    }

    pub fn index(&self, #[comptime] i: u32) -> E {
        self.vals.index(i).val
    }

    pub fn copy(&self) -> RowWise<E> {
        let mut vals = Sequence::new();
        #[unroll]
        for i in 0..self.num_rows {
            vals.push(*self.vals.index(i));
        }
        RowWise::<E>::new(self.num_rows, vals)
    }

    pub fn copy_into(&self, other: &mut RowWise<E>) {
        #[unroll]
        for i in 0..self.num_rows {
            let place = other.vals.index_mut(i);
            let value = self.vals.index(i);
            place.val = value.val;
        }
    }

    pub fn cast<E2: Float>(&self) -> RowWise<E2> {
        let mut vals = Sequence::new();
        #[unroll]
        for i in 0..self.num_rows {
            vals.push(self.vals.index(i).cast::<E2>());
        }
        RowWise::<E2>::new(self.num_rows, vals)
    }
}

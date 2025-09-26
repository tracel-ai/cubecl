use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::TileMask;
use crate::components::tile::dummy::FlashPrecision;

#[cube]
pub trait SoftmaxTile<FP: FlashPrecision>: CubeType {
    type Fragment: SoftmaxFragment<FP::SP>;

    fn init_state() -> RunningState<FP::SP>;

    fn zero(&mut self);

    fn fragment(&mut self) -> &mut Self::Fragment;

    fn scale_and_mask(&mut self, scale: FP::SP, mask: TileMask);

    fn row_max(&self, base: RowWise<FP::SP>) -> RowWise<FP::SP>;

    /// Converts scores → probabilities, updates running state,
    /// and returns the factor needed to scale the accumulator
    fn to_prob(
        &mut self,
        state: &mut RunningState<FP::SP>,
        max: &RowWise<FP::SP>,
    ) -> RowWise<FP::A>;
}

#[derive(CubeType)]
pub struct RowWise<E: Float> {
    #[cube(comptime)]
    num_rows: u32,
    vals: Sequence<RowVal<E>>,
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
}

#[cube]
impl<E: Float> RowWise<E> {
    pub fn new(#[comptime] num_rows: u32, vals: Sequence<RowVal<E>>) -> RowWise<E> {
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
}

#[derive(CubeType)]
pub struct RunningState<E: Float> {
    pub m: RowWise<E>,
    pub l: RowWise<E>,
}

#[cube]
impl<E: Float> RunningState<E> {
    pub fn init(#[comptime] num_rows: u32) -> RunningState<E> {
        let mut m = Sequence::new();
        let mut l = Sequence::new();
        #[unroll]
        for _ in 0..num_rows {
            m.push(RowVal::new(E::from_int(-99999999999)));
            l.push(RowVal::new(E::from_int(0)));
        }

        RunningState::<E> {
            m: RowWise::<E>::new(num_rows, m),
            l: RowWise::<E>::new(num_rows, l),
        }
    }

    pub fn update(&mut self, new_m: RowWise<E>, new_l: RowWise<E>) {
        new_m.copy_into(&mut self.m);
        new_l.copy_into(&mut self.l);
    }
}

#[cube]
pub trait SoftmaxFragment<E: Float>: CubeType {}

#[derive(CubeType)]
pub struct RowStats<E: Float> {
    pub score_max: RowWise<E>,
    pub prob_sum: RowWise<E>,
}

#[cube]
impl<E: Float> RowStats<E> {
    pub fn new(score_max: RowWise<E>, prob_sum: RowWise<E>) -> RowStats<E> {
        RowStats::<E> {
            score_max,
            prob_sum,
        }
    }
}

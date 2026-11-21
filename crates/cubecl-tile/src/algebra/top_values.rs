//! The `K` best values seen, without the indices they were read at, for a top-k that writes
//! only its values.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::algebra::indexed::{nan_lanes, ordered_only, same_value};
use crate::algebra::top_k::TOP_K_UNROLL_BUDGET;
use crate::space::UnitShare;
use crate::{Carrier, Monoid};

/// The `K` best values seen, best first, pointwise on vectors: `Monoid::Max` over it is the
/// top-k's values, `Monoid::Min` the bottom-k's.
///
/// [`TopK`](crate::TopK) without the indices: each step of a walk is one compare where `TopK`'s
/// is a compare of values, then of indices, and it moves half the registers. Best is the larger
/// value under `Max` (smaller under `Min`), a NaN ahead of every number; equal values are
/// interchangeable, so which of them is kept does not show. The identity is `K` of the monoid's
/// identity.
///
/// `Sum` and `Prod` are refused when the kernel expands.
#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct TopValues<E: Numeric, N: Size, K: Size> {
    pub values: Array<Vector<E, N>>,
    #[cube(comptime)]
    k: PhantomData<K>,
}

#[cube]
impl<E: Numeric, N: Size, K: Size> TopValues<E, N, K> {
    /// The list holding `values`, best first: a list stored apart and read back.
    pub fn new(values: Array<Vector<E, N>>) -> Self {
        TopValues::<E, N, K> {
            values,
            k: comptime!(PhantomData),
        }
    }

    /// `value` kept in this list if it is among the `K` best, in place. A value the last kept
    /// slot stays ahead of on every lane changes nothing and skips the walk: over a long row,
    /// almost every value. The walk unrolls within [`TOP_K_UNROLL_BUDGET`]; past it, a one-lane
    /// list shifts the value in from the end instead, as [`TopK::insert`](crate::TopK::insert)
    /// does.
    pub fn insert(&mut self, value: Vector<E, N>, #[comptime] monoid: Monoid) {
        ordered_only(monoid);
        let k = K::value();
        let lanes = N::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        if !every_lane::<N>(stays_ahead::<E, N>(self.values[k - 1], value, monoid)) {
            if comptime!(unrolled || lanes > 1) {
                self.walk(value, monoid, unrolled);
            } else {
                self.shift_in(value, monoid);
            }
        }
    }

    /// `value` walked down this list: each slot keeps the better of what it holds and what comes
    /// down, and passes the other on, picked before the slot is written as in `TopK`'s walk.
    fn walk(
        &mut self,
        value: Vector<E, N>,
        #[comptime] monoid: Monoid,
        #[comptime] unrolled: bool,
    ) {
        let k = K::value();
        let mut value = value;
        #[unroll(unrolled)]
        for j in 0..k {
            let kept = self.values[j];
            let keep = stays_ahead::<E, N>(kept, value, monoid);
            let passed = select_many(keep, value, kept);
            self.values[j] = select_many(keep, kept, value);
            value = passed;
        }
    }

    /// `value`, entering this one-lane list past its last slot: the slots it beats move down
    /// one, from the end up, and it takes the last of them. A value that enters a long list
    /// lands near its end, so this touches a few slots where a walk touches every one.
    fn shift_in(&mut self, value: Vector<E, N>, #[comptime] monoid: Monoid) {
        let k = K::value();
        let mut slot = comptime!(k - 1).runtime();
        loop {
            if slot == 0 {
                break;
            }
            let above = slot - 1;
            if every_lane::<N>(stays_ahead::<E, N>(self.values[above], value, monoid)) {
                break;
            }
            self.values[slot] = self.values[above];
            slot = above;
        }
        self.values[slot] = value;
    }

    /// The `K` best of the plane's lists, in `K` rounds: the holders' heads meet, the winner
    /// takes the round's slot, and the lowest unit that held it moves its list up one. For lists
    /// in registers, unrolled: a move is a select per slot.
    fn across_plane_by_shifts(value: Self, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let none = Vector::<E, N>::new(Monoid::identity::<E>(monoid));
        let unit = Vector::<u32, N>::new(UNIT_POS_PLANE);
        let mut held = Array::new(k);
        #[unroll]
        for j in 0..k {
            held[j] = value.values[j];
        }
        let mut values = Array::new(k);
        #[unroll(unrolled)]
        for round in 0..k {
            let head = held[0];
            let best = Vector::<E, N>::across(head, UnitShare::Plane, monoid);
            values[round] = best;
            let holder = plane_min(select_many(
                same_value::<E, N>(head, best),
                unit,
                Vector::new(u32::MAX),
            ));
            let taken = unit.equal(&holder);
            #[unroll(unrolled)]
            for j in 0..comptime!(k - 1) {
                held[j] = select_many(taken, held[j + 1], held[j]);
            }
            held[k - 1] = select_many(taken, none, held[k - 1]);
        }
        TopValues::<E, N, K> {
            values,
            k: comptime!(PhantomData),
        }
    }

    /// The `K` best of the plane's one-lane lists, in `K` rounds as
    /// [`across_plane_by_shifts`](Self::across_plane_by_shifts) does, but each unit keeps its
    /// head apart and the lowest unit that held the winner reads its next at a cursor: for lists
    /// too long to unroll, which live in memory, where moving a whole list each round costs `K`
    /// loads and stores.
    fn across_plane_by_cursor(value: Self, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unit = Vector::<u32, N>::new(UNIT_POS_PLANE);
        let mut values = Array::new(k);
        let none = Vector::<E, N>::new(Monoid::identity::<E>(monoid));
        let mut head = value.values[0];
        let mut cursor = comptime!(0usize).runtime();
        for round in 0..k {
            let best = Vector::<E, N>::across(head, UnitShare::Plane, monoid);
            values[round] = best;
            let holder = plane_min(select_many(
                same_value::<E, N>(head, best),
                unit,
                Vector::new(u32::MAX),
            ));
            if every_lane::<N>(unit.equal(&holder)) {
                cursor += 1;
                head = none;
                if cursor < k {
                    head = value.values[cursor];
                }
            }
        }
        TopValues::<E, N, K> {
            values,
            k: comptime!(PhantomData),
        }
    }
}

/// Where `kept` stays ahead of `value`: it is better, as good, or NaN.
#[cube]
fn stays_ahead<E: Numeric, N: Size>(
    kept: Vector<E, N>,
    value: Vector<E, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    let elem = elem_type_of::<E>();
    // `value` is not better: the negated compare, so that a NaN `value` is better.
    let by_order = match monoid {
        Monoid::Max => kept.greater_equal(&value),
        _ => kept.less_equal(&value),
    };
    if comptime!(elem.is_float()) {
        by_order.or(nan_lanes::<E, N>(kept))
    } else {
        by_order
    }
}

/// Whether every lane of `lanes` is set.
#[cube]
fn every_lane<N: Size>(lanes: Vector<bool, N>) -> bool {
    let mut every = true;
    #[unroll]
    for lane in 0..N::value() {
        if !lanes.extract(lane) {
            every = false;
        }
    }
    every
}

#[cube]
impl<E: Numeric, N: Size, K: Size> Carrier for TopValues<E, N, K> {
    fn combine(lhs: Self, rhs: Self, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut total = lhs;
        #[unroll(unrolled)]
        for j in 0..k {
            total.insert(rhs.values[j], monoid);
        }
        total
    }

    fn identity(#[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let none = Vector::<E, N>::new(Monoid::identity::<E>(monoid));
        let mut values = Array::new(k);
        #[unroll]
        for j in 0..k {
            values[j] = none;
        }
        TopValues::<E, N, K> {
            values,
            k: comptime!(PhantomData),
        }
    }

    /// Across a plane, `K` rounds: the holders' heads meet, the winner takes the round's slot,
    /// and the lowest unit that held it moves on to its next: by moving its list up one where
    /// the list unrolls, by a cursor where a one-lane list is too long to. Across groups, the
    /// lists are swapped and combined, one unit bit at a time.
    fn across(value: Self, #[comptime] share: UnitShare, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let hardware = comptime::hardware_properties().comptime();
        // A singleton plane already holds the entire list and needs no lane communication.
        let share = comptime!(
            if hardware.plane_size_min == 1 && hardware.plane_size_max == 1 {
                UnitShare::Whole
            } else {
                share
            }
        );
        match share {
            UnitShare::Repeated | UnitShare::Whole => value,
            UnitShare::Plane => {
                let k = K::value();
                let lanes = N::value();
                if comptime!(k * k <= TOP_K_UNROLL_BUDGET || lanes > 1) {
                    TopValues::<E, N, K>::across_plane_by_shifts(value, monoid)
                } else {
                    TopValues::<E, N, K>::across_plane_by_cursor(value, monoid)
                }
            }
            UnitShare::Group { unit_bits } => {
                let k = K::value();
                let mut total = value;
                #[unroll]
                for bit in 0..comptime!(usize::BITS - unit_bits.leading_zeros()) {
                    if comptime!(unit_bits & (1 << bit) != 0) {
                        let mask = 1u32 << bit;
                        let mut other = Array::new(k);
                        #[unroll]
                        for j in 0..k {
                            other[j] = plane_shuffle_xor(total.values[j], mask);
                        }
                        total = TopValues::<E, N, K>::combine(
                            total,
                            TopValues::<E, N, K>::new(other),
                            monoid,
                        );
                    }
                }
                total
            }
        }
    }
}

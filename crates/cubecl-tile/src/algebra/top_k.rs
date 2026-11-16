//! The `K` best values seen, each with the index it was read at, for the top-k.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::algebra::indexed::{nan_lanes, ordered_only, wins};
use crate::space::UnitShare;
use crate::{Carrier, Indexed, Monoid};

/// The most steps a `K`-by-`K` walk of a [`TopK`] is unrolled to: past it, as for the `k = 300`
/// of an object-detection head, the walks stay loops, since unrolled the kernel grows with `K²`
/// and the backend compilers stop returning from it.
pub const TOP_K_UNROLL_BUDGET: usize = 1024;

/// The `K` best values seen, each with the index it was read at, best first, pointwise on
/// vectors: `Monoid::Max` over it is the top-k, `Monoid::Min` the bottom-k.
///
/// Best is [`Indexed`]'s order: the larger value under `Max` (smaller under `Min`), then the lower
/// index, a NaN ahead of every number. Combining two lists keeps the `K` best of both, which is
/// associative and commutative, and the identity is `K` of [`Indexed`]'s identity. A list holds
/// each index once, as a reduction reads each index once; the plane step finds the unit that
/// held a winner by its index.
///
/// `Sum` and `Prod` are refused when the kernel expands, as on [`Indexed`].
#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct TopK<E: Numeric, N: Size, K: Size> {
    pub values: Array<Vector<E, N>>,
    pub indices: Array<Vector<u32, N>>,
    #[cube(comptime)]
    k: PhantomData<K>,
}

#[cube]
impl<E: Numeric, N: Size, K: Size> TopK<E, N, K> {
    /// The list holding `values`, best first, each read at its index in `indices`: a list stored
    /// apart and read back.
    pub fn new(values: Array<Vector<E, N>>, indices: Array<Vector<u32, N>>) -> Self {
        TopK::<E, N, K> {
            values,
            indices,
            k: comptime!(PhantomData),
        }
    }

    /// `value`, read at `index`, kept in this list if it is among the `K` best: the one-item
    /// combine, `K` steps where combining with a list of one is `K * K`. In place, since a
    /// reduction inserts once a value read: a list returned anew is copied back slot by slot.
    ///
    /// A value that is no NaN on any lane, almost every value, walks with the compares numbers
    /// take: it moves into a slot it beats by value, or ties by value at a lower index, and a NaN
    /// slot, which compares false, stays ahead of it as the order wants. It skips the walk where
    /// it beats the last kept slot on no lane: over a long row, almost every value. Its walk
    /// unrolls within [`TOP_K_UNROLL_BUDGET`]. A value with a NaN, rare, walks in the full order,
    /// and that walk stays a loop: unrolled beside the first, it would double the walks' code.
    /// For a list in registers, the loop turns the list instead of picking a slot at run time
    /// ([`walk_turning`](Self::walk_turning)).
    pub fn insert(
        &mut self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
    ) {
        ordered_only(monoid);
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let elem = elem_type_of::<E>();
        let mut nan = false;
        if comptime!(elem.is_float()) {
            nan = any_lane::<N>(nan_lanes::<E, N>(value));
        }
        if nan {
            if unrolled {
                self.walk_turning(value, index, monoid);
            } else {
                self.walk_in_full_order(value, index, monoid, false);
            }
        } else {
            let enters = takes_the_slot::<E, N>(
                self.values[k - 1],
                self.indices[k - 1],
                value,
                index,
                monoid,
            );
            let lanes = N::value();
            if any_lane::<N>(enters) {
                if comptime!(unrolled || lanes > 1) {
                    self.walk(value, index, monoid, unrolled);
                } else {
                    self.shift_in(value, index, monoid);
                }
            }
        }
    }

    /// `value`, read at `index` and no NaN, walked down this list: each slot keeps the better
    /// of what it holds and what comes down, and passes the other on.
    ///
    /// What a slot passes on is picked before the slot is written. Metal on Apple GPUs keeps a
    /// list walked the other way round out of registers, and a top-32 runs twice as long.
    fn walk(
        &mut self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
        #[comptime] unrolled: bool,
    ) {
        let k = K::value();
        let mut value = value;
        let mut index = index;
        #[unroll(unrolled)]
        for j in 0..k {
            let kept_value = self.values[j];
            let kept_index = self.indices[j];
            let takes = takes_the_slot::<E, N>(kept_value, kept_index, value, index, monoid);
            let passed_value = select_many(takes, kept_value, value);
            self.values[j] = select_many(takes, value, kept_value);
            value = passed_value;
            let passed_index = select_many(takes, kept_index, index);
            self.indices[j] = select_many(takes, index, kept_index);
            index = passed_index;
        }
    }

    /// `value`, read at `index`, walked down this list in the full order as
    /// [`walk_in_full_order`](Self::walk_in_full_order) does, but as a loop that touches fixed
    /// slots only: each step meets the first slot, then the list turns up one slot, and after
    /// `K` steps every slot is back in place. For a list in registers, which a slot picked at run
    /// time would move to memory for the whole kernel: the walk of a rare NaN read.
    fn walk_turning(
        &mut self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
    ) {
        let k = K::value();
        let mut value = value;
        let mut index = index;
        for _step in 0..k {
            let kept = Indexed::<E, N> {
                value: self.values[0],
                index: self.indices[0],
            };
            let keep = wins::<E, N>(kept, Indexed::<E, N> { value, index }, monoid);
            let passed_value = select_many(keep, value, kept.value);
            let passed_index = select_many(keep, index, kept.index);
            #[unroll]
            for j in 0..comptime!(k - 1) {
                self.values[j] = self.values[j + 1];
                self.indices[j] = self.indices[j + 1];
            }
            self.values[k - 1] = select_many(keep, kept.value, value);
            self.indices[k - 1] = select_many(keep, kept.index, index);
            value = passed_value;
            index = passed_index;
        }
    }

    /// `value`, read at `index` and no NaN, entering this one-lane list past its last slot: the
    /// slots it beats move down one, from the end up, and it takes the last of them. A value
    /// that enters a long list lands near its end, so this touches a few slots where a walk
    /// touches every one, and a list too long to unroll lives in memory, where each touch costs.
    fn shift_in(&mut self, value: Vector<E, N>, index: Vector<u32, N>, #[comptime] monoid: Monoid) {
        let k = K::value();
        let mut slot = comptime!(k - 1).runtime();
        loop {
            if slot == 0 {
                break;
            }
            let above = slot - 1;
            let takes = takes_the_slot::<E, N>(
                self.values[above],
                self.indices[above],
                value,
                index,
                monoid,
            );
            if any_lane::<N>(takes) {
                self.values[slot] = self.values[above];
                self.indices[slot] = self.indices[above];
                slot = above;
            } else {
                break;
            }
        }
        self.values[slot] = value;
        self.indices[slot] = index;
    }

    /// The `K` best of the units' lists, in `K` rounds: the holders' heads meet as an
    /// [`Indexed`] does, the winner takes the round's slot, and the unit that held it moves its
    /// list up one. For lists in registers, unrolled: a move is a select per slot.
    fn across_by_shifts(
        value: Self,
        #[comptime] share: UnitShare,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let none = Indexed::<E, N>::identity(monoid);
        let mut held_values = Array::new(k);
        let mut held_indices = Array::new(k);
        #[unroll]
        for j in 0..k {
            held_values[j] = value.values[j];
            held_indices[j] = value.indices[j];
        }
        let mut values = Array::new(k);
        let mut indices = Array::new(k);
        #[unroll(unrolled)]
        for round in 0..k {
            let head = Indexed::<E, N> {
                value: held_values[0],
                index: held_indices[0],
            };
            let best = Indexed::<E, N>::across(head, share, monoid);
            values[round] = best.value;
            indices[round] = best.index;
            let taken = held_indices[0].equal(&best.index);
            #[unroll(unrolled)]
            for j in 0..comptime!(k - 1) {
                held_values[j] = select_many(taken, held_values[j + 1], held_values[j]);
                held_indices[j] = select_many(taken, held_indices[j + 1], held_indices[j]);
            }
            held_values[k - 1] = select_many(taken, none.value, held_values[k - 1]);
            held_indices[k - 1] = select_many(taken, none.index, held_indices[k - 1]);
        }
        TopK::<E, N, K> {
            values,
            indices,
            k: comptime!(PhantomData),
        }
    }

    /// The `K` best of the units' one-lane lists, in `K` rounds as
    /// [`across_by_shifts`](Self::across_by_shifts) does, but each unit keeps its head apart and
    /// the unit that held the winner reads its next at a cursor: for lists too long to unroll,
    /// which live in memory, where moving a whole list each round costs `K` loads and stores.
    fn across_by_cursor(
        value: Self,
        #[comptime] share: UnitShare,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let k = K::value();
        let mut values = Array::new(k);
        let mut indices = Array::new(k);
        let none = Indexed::<E, N>::identity(monoid);
        let mut head = Indexed::<E, N> {
            value: value.values[0],
            index: value.indices[0],
        };
        let mut cursor = comptime!(0usize).runtime();
        for round in 0..k {
            let best = Indexed::<E, N>::across(head, share, monoid);
            values[round] = best.value;
            indices[round] = best.index;
            if any_lane::<N>(head.index.equal(&best.index)) {
                cursor += 1;
                head = none;
                if cursor < k {
                    head = Indexed::<E, N> {
                        value: value.values[cursor],
                        index: value.indices[cursor],
                    };
                }
            }
        }
        TopK::<E, N, K> {
            values,
            indices,
            k: comptime!(PhantomData),
        }
    }

    /// `value`, read at `index`, kept in this list if it is among the `K` best, every step
    /// comparing in the full order (value, then the lower index, NaN first): for values that
    /// may be NaN, such as another list's. One walk, where [`insert`](Self::insert) has two,
    /// so an unrolled combine stays half the size. A value the last kept slot beats on every
    /// lane skips the walk.
    pub fn insert_in_full_order(
        &mut self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
    ) {
        ordered_only(monoid);
        let k = K::value();
        let last = Indexed::<E, N> {
            value: self.values[k - 1],
            index: self.indices[k - 1],
        };
        let read = Indexed::<E, N> { value, index };
        if any_lane::<N>(wins::<E, N>(read, last, monoid)) {
            self.walk_in_full_order(
                value,
                index,
                monoid,
                comptime!(k * k <= TOP_K_UNROLL_BUDGET),
            );
        }
    }

    /// `value`, read at `index`, walked down this list in the full order: each slot keeps the
    /// better of what it holds and what comes down, and passes the other on, picked before the
    /// slot is written as in [`walk`](Self::walk).
    fn walk_in_full_order(
        &mut self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
        #[comptime] unrolled: bool,
    ) {
        let k = K::value();
        let mut value = value;
        let mut index = index;
        #[unroll(unrolled)]
        for j in 0..k {
            let kept = Indexed::<E, N> {
                value: self.values[j],
                index: self.indices[j],
            };
            let keep = wins::<E, N>(kept, Indexed::<E, N> { value, index }, monoid);
            let passed_value = select_many(keep, value, kept.value);
            self.values[j] = select_many(keep, kept.value, value);
            value = passed_value;
            let passed_index = select_many(keep, index, kept.index);
            self.indices[j] = select_many(keep, kept.index, index);
            index = passed_index;
        }
    }
}

/// Where `value`, read at `index` and no NaN, takes the slot holding `kept` read at
/// `kept_index`: it is better, or as good at a lower index. A NaN `kept` compares false both
/// ways, so it keeps its slot.
#[cube]
fn takes_the_slot<E: Numeric, N: Size>(
    kept: Vector<E, N>,
    kept_index: Vector<u32, N>,
    value: Vector<E, N>,
    index: Vector<u32, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    let better = match monoid {
        Monoid::Max => kept.less_than(&value),
        _ => kept.greater_than(&value),
    };
    select_many(kept.equal(&value), index.less_than(&kept_index), better)
}

/// Whether any lane of `lanes` is set.
#[cube]
fn any_lane<N: Size>(lanes: Vector<bool, N>) -> bool {
    let mut any = false;
    #[unroll]
    for lane in 0..N::value() {
        if lanes.extract(lane) {
            any = true;
        }
    }
    any
}

#[cube]
impl<E: Numeric, N: Size, K: Size> Carrier for TopK<E, N, K> {
    fn combine(lhs: Self, rhs: Self, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut total = lhs;
        #[unroll(unrolled)]
        for j in 0..k {
            total.insert_in_full_order(rhs.values[j], rhs.indices[j], monoid);
        }
        total
    }

    fn identity(#[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let none = Indexed::<E, N>::identity(monoid);
        let mut values = Array::new(k);
        let mut indices = Array::new(k);
        #[unroll]
        for j in 0..k {
            values[j] = none.value;
            indices[j] = none.index;
        }
        TopK::<E, N, K> {
            values,
            indices,
            k: comptime!(PhantomData),
        }
    }

    /// `K` rounds: the holders' heads meet as an [`Indexed`] does, the winner takes the round's
    /// slot, and the unit that held it moves on to its next: by moving its list up one where the
    /// list unrolls, by a cursor where a one-lane list is too long to.
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
            UnitShare::Plane | UnitShare::Group { .. } => {
                let k = K::value();
                let lanes = N::value();
                if comptime!(k * k <= TOP_K_UNROLL_BUDGET || lanes > 1) {
                    TopK::<E, N, K>::across_by_shifts(value, share, monoid)
                } else {
                    TopK::<E, N, K>::across_by_cursor(value, share, monoid)
                }
            }
        }
    }
}

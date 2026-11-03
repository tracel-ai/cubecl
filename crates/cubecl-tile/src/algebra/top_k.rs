//! The `K` best values seen, each with the index it was read at, for the top-k.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::algebra::indexed::{ordered_only, wins};
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

    /// This list with `value`, read at `index`, kept if it is among the `K` best: the one-item
    /// combine, `K` steps where combining with a list of one is `K * K`. Its walk unrolls
    /// within [`TOP_K_UNROLL_BUDGET`], as every walk of a `K`-by-`K` nest does.
    pub fn insert(
        self,
        value: Vector<E, N>,
        index: Vector<u32, N>,
        #[comptime] monoid: Monoid,
    ) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let mut values = Array::new(k);
        let mut indices = Array::new(k);
        let mut value = value;
        let mut index = index;
        #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
        for j in 0..k {
            let kept = Indexed::<E, N> {
                value: self.values[j],
                index: self.indices[j],
            };
            let keep = wins::<E, N>(kept, Indexed::<E, N> { value, index }, monoid);
            values[j] = select_many(keep, kept.value, value);
            indices[j] = select_many(keep, kept.index, index);
            value = select_many(keep, value, kept.value);
            index = select_many(keep, index, kept.index);
        }
        TopK::<E, N, K> {
            values,
            indices,
            k: comptime!(PhantomData),
        }
    }
}

#[cube]
impl<E: Numeric, N: Size, K: Size> Carrier for TopK<E, N, K> {
    fn combine(lhs: Self, rhs: Self, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let k = K::value();
        let mut total = lhs;
        #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
        for j in 0..k {
            total = total.insert(rhs.values[j], rhs.indices[j], monoid);
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
    /// slot, and the unit that held it moves its list up one.
    fn across(value: Self, #[comptime] share: UnitShare, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        match comptime!(share) {
            UnitShare::Repeated | UnitShare::Whole => value,
            UnitShare::Plane | UnitShare::Group { .. } => {
                let k = K::value();
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
                #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
                for round in 0..k {
                    let head = Indexed::<E, N> {
                        value: held_values[0],
                        index: held_indices[0],
                    };
                    let best = Indexed::<E, N>::across(head, share, monoid);
                    values[round] = best.value;
                    indices[round] = best.index;
                    let taken = held_indices[0].equal(&best.index);
                    #[unroll(k * k <= TOP_K_UNROLL_BUDGET)]
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
        }
    }
}

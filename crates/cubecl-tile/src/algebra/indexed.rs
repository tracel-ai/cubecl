//! A value carried with the index it was read at, for the argmax and argmin.

use cubecl::ir::dialect::math::IsNanOp;
use cubecl::prelude::*;

use crate::space::UnitShare;
use crate::{Carrier, Monoid};

/// A value with the index it was read at, pointwise on vectors: `Monoid::Max` over it is the
/// argmax, `Monoid::Min` the argmin.
///
/// Under `Max` the larger value wins, under `Min` the smaller; between equal values, the lower
/// index. A NaN wins over every number under both, as the monoid keeps NaN on numbers, and
/// between NaNs the lower index wins: the result is the first NaN's index. `-0` and `+0` are
/// equal values. The identities are `(-inf, u32::MAX)` under `Max` and `(+inf, u32::MAX)` under
/// `Min`, an integer's bounds in place of the infinities, which every read value beats, `-inf`
/// itself by its lower index.
///
/// `Sum` and `Prod` have no meaning on an index, so they are refused when the kernel expands.
#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Indexed<E: Numeric, N: Size> {
    pub value: Vector<E, N>,
    pub index: Vector<u32, N>,
}

#[cube]
impl<E: Numeric, N: Size> Indexed<E, N> {
    /// [`combine`](Carrier::combine) where `later` was read at a higher index than any value
    /// `earlier` holds, as a unit's reads along a row are: a tie then keeps `earlier`, unless it
    /// holds no read yet (the identity, at index `u32::MAX`), so `later` wins where it is ahead
    /// by value, NaN where `earlier` is not, or `earlier` is the identity. Three compares a lane,
    /// against about ten for the full order, which a read loop pays on every value.
    pub fn combine_later(earlier: Self, later: Self, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let takes = later_takes::<E, N>(earlier.value, earlier.index, later.value, monoid);
        Indexed::<E, N> {
            value: select_many(takes, later.value, earlier.value),
            index: select_many(takes, later.index, earlier.index),
        }
    }
}

/// Where `later`, read at a higher index than `held` was read at (`held_index`), beats it: past
/// it in the monoid's order or NaN, while `held` is no NaN; or anywhere `held` is no read yet
/// (the identity, at index `u32::MAX`). Not being behind or level is being ahead or NaN, one
/// unordered compare, and only a NaN is unequal to itself, which an integer never is.
#[cube(harness)]
fn later_takes<E: Numeric, N: Size>(
    held: Vector<E, N>,
    held_index: Vector<u32, N>,
    later: Vector<E, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    let behind_or_level = match monoid {
        Monoid::Max => later.less_equal(&held),
        _ => later.greater_equal(&held),
    };
    let ahead_or_nan = !behind_or_level;
    let holds_no_read = held_index.equal(&Vector::new(u32::MAX));
    ahead_or_nan.vec_and(held.equal(&held)).or(holds_no_read)
}

#[cube]
impl<E: Numeric, N: Size> Carrier for Indexed<E, N> {
    fn combine(lhs: Self, rhs: Self, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        let keep_lhs = wins::<E, N>(lhs, rhs, monoid);
        Indexed::<E, N> {
            value: select_many(keep_lhs, lhs.value, rhs.value),
            index: select_many(keep_lhs, lhs.index, rhs.index),
        }
    }

    fn identity(#[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        Indexed::<E, N> {
            value: Vector::new(Monoid::identity::<E>(monoid)),
            index: Vector::new(u32::MAX),
        }
    }

    fn across(value: Self, #[comptime] share: UnitShare, #[comptime] monoid: Monoid) -> Self {
        ordered_only(monoid);
        match share {
            UnitShare::Repeated | UnitShare::Whole => value,
            // The winning value, then the lowest index among the units that hold it.
            UnitShare::Plane => {
                let best = Vector::<E, N>::across(value.value, share, monoid);
                let holder = select_many(
                    same_value::<E, N>(value.value, best),
                    value.index,
                    Vector::new(u32::MAX),
                );
                Indexed::<E, N> {
                    value: best,
                    index: plane_min(holder),
                }
            }
            UnitShare::Group { unit_bits } => {
                let mut total = value;
                #[unroll]
                for bit in 0..comptime!(usize::BITS - unit_bits.leading_zeros()) {
                    if comptime!(unit_bits & (1 << bit) != 0) {
                        let mask = 1u32 << bit;
                        let other = Indexed::<E, N> {
                            value: plane_shuffle_xor(total.value, mask),
                            index: plane_shuffle_xor(total.index, mask),
                        };
                        total = Indexed::<E, N>::combine(total, other, monoid);
                    }
                }
                total
            }
        }
    }
}

/// Where `lhs` wins over `rhs` under `monoid`: by value, then by the lower index.
#[cube]
pub(crate) fn wins<E: Numeric, N: Size>(
    lhs: Indexed<E, N>,
    rhs: Indexed<E, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    select_many(
        same_value::<E, N>(lhs.value, rhs.value),
        lhs.index.less_than(&rhs.index),
        ahead::<E, N>(lhs.value, rhs.value, monoid),
    )
}

/// Refuses `Sum` and `Prod`, which an index has no meaning under.
#[cube]
pub(crate) fn ordered_only(#[comptime] monoid: Monoid) {
    comptime!(assert!(
        matches!(monoid, Monoid::Max | Monoid::Min),
        "Indexed: {monoid:?} has no meaning on an index; an Indexed combines under Max or Min"
    ));
}

/// Where `lhs` wins over `rhs` by value alone: larger under `Max`, smaller under `Min`, or NaN.
#[cube(harness)]
fn ahead<E: Numeric, N: Size>(
    lhs: Vector<E, N>,
    rhs: Vector<E, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    let elem = elem_type_of::<E>();
    let by_order = match monoid {
        Monoid::Max => lhs.greater_than(&rhs),
        _ => lhs.less_than(&rhs),
    };
    if comptime!(elem.is_float()) {
        nan_lanes::<E, N>(lhs).or(by_order)
    } else {
        by_order
    }
}

/// Where `lhs` and `rhs` tie by value: equal, or both NaN.
#[cube(harness)]
pub(crate) fn same_value<E: Numeric, N: Size>(
    lhs: Vector<E, N>,
    rhs: Vector<E, N>,
) -> Vector<bool, N> {
    let elem = elem_type_of::<E>();
    if comptime!(elem.is_float()) {
        lhs.equal(&rhs)
            .or(nan_lanes::<E, N>(lhs).vec_and(nan_lanes::<E, N>(rhs)))
    } else {
        lhs.equal(&rhs)
    }
}

/// Where `value` is NaN. `cubecl`'s `is_nan` asks the type for a float element; this one takes any
/// element and is called only once the element is known to be a float.
#[cube]
pub(crate) fn nan_lanes<E: Numeric, N: Size>(value: Vector<E, N>) -> Vector<bool, N> {
    intrinsic!(|scope| {
        let value = value.read_value(scope);
        let op = IsNanOp::new(scope.ctx_mut(), value);
        scope.register_with_result(&op).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate cubecl as cubecl_runtime;
    use cubecl_runtime::TestRuntime;

    /// Larger, smaller, a NaN against a number, and equal: under `Max` the larger and the NaN win.
    #[test]
    fn ahead_under_max_is_larger_or_nan() {
        let ahead = ahead::harness::<TestRuntime, f32, Const<4>>(
            &[2.0, 1.0, f32::NAN, 3.0],
            &[1.0, 2.0, 0.0, 3.0],
            Monoid::Max,
        );
        assert_eq!(ahead, [true, false, true, false]);
    }

    /// Larger, smaller, a NaN against a number, and equal: under `Min` the smaller and the NaN win.
    #[test]
    fn ahead_under_min_is_smaller_or_nan() {
        let ahead = ahead::harness::<TestRuntime, f32, Const<4>>(
            &[2.0, 1.0, f32::NAN, 3.0],
            &[1.0, 2.0, 0.0, 3.0],
            Monoid::Min,
        );
        assert_eq!(ahead, [false, true, true, false]);
    }

    /// Against a held read: a later larger value, a later equal one, a later NaN, and any later
    /// value against a held NaN. Under `Max` the larger and the NaN take over.
    #[test]
    fn under_max_a_later_larger_value_or_nan_takes_over() {
        let takes = later_takes::harness::<TestRuntime, f32, Const<4>>(
            &[1.0, 2.0, 2.0, f32::NAN],
            &[0, 0, 0, 0],
            &[2.0, 2.0, f32::NAN, 3.0],
            Monoid::Max,
        );
        assert_eq!(takes, [true, false, true, false]);
    }

    /// Against a held read: a later smaller value, a later equal one, a later NaN, and any later
    /// value against a held NaN. Under `Min` the smaller and the NaN take over.
    #[test]
    fn under_min_a_later_smaller_value_or_nan_takes_over() {
        let takes = later_takes::harness::<TestRuntime, f32, Const<4>>(
            &[2.0, 2.0, 2.0, f32::NAN],
            &[0, 0, 0, 0],
            &[1.0, 2.0, f32::NAN, 0.0],
            Monoid::Min,
        );
        assert_eq!(takes, [true, false, true, false]);
    }

    /// The identity, which holds no read, is taken over by any read: one level with it, one
    /// behind it in the monoid's order, and a NaN.
    #[test]
    fn any_read_takes_over_the_identity() {
        let takes = later_takes::harness::<TestRuntime, f32, Const<3>>(
            &[f32::NEG_INFINITY; 3],
            &[u32::MAX; 3],
            &[f32::NEG_INFINITY, 0.0, f32::NAN],
            Monoid::Max,
        );
        assert_eq!(takes, [true, true, true]);
    }

    /// Equal numbers, two NaNs, a NaN against a number, and different numbers.
    #[test]
    fn same_value_is_equal_or_both_nan() {
        let same = same_value::harness::<TestRuntime, f32, Const<4>>(
            &[1.0, f32::NAN, f32::NAN, 2.0],
            &[1.0, f32::NAN, 0.0, 3.0],
        );
        assert_eq!(same, [true, true, false, false]);
    }
}

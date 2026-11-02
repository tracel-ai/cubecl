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
/// `Min`, which every read value beats, `-inf` itself by its lower index.
///
/// `Sum` and `Prod` have no meaning on an index, so they are refused when the kernel expands.
#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Indexed<E: Numeric, N: Size> {
    pub value: Vector<E, N>,
    pub index: Vector<u32, N>,
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
        match comptime!(share) {
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
                        let mask = comptime!(1u32 << bit);
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
#[cube]
fn ahead<E: Numeric, N: Size>(
    lhs: Vector<E, N>,
    rhs: Vector<E, N>,
    #[comptime] monoid: Monoid,
) -> Vector<bool, N> {
    let elem = elem_type_of::<E>();
    let by_order = match comptime!(monoid) {
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
#[cube]
fn same_value<E: Numeric, N: Size>(lhs: Vector<E, N>, rhs: Vector<E, N>) -> Vector<bool, N> {
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
fn nan_lanes<E: Numeric, N: Size>(value: Vector<E, N>) -> Vector<bool, N> {
    intrinsic!(|scope| {
        let value = value.read_value(scope);
        let op = IsNanOp::new(scope.ctx_mut(), value);
        scope.register_with_result(&op).into()
    })
}

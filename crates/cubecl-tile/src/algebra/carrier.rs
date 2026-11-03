//! What a [`Monoid`] combines: numbers, and the carriers built over them.

use cubecl::ir::dialect::cmp::{FMaxNanOp, FMinNanOp};
use cubecl::prelude::*;

use crate::Monoid;
use crate::space::UnitShare;

/// A value a [`Monoid`] combines, and how each monoid combines it. A monoid is a carrier, an
/// operation and an identity, so the carrier says what each monoid it has does on it; a carrier
/// without one refuses it when the kernel expands.
///
/// Numbers and their vectors have all four monoids. [`Indexed`](crate::Indexed) and
/// [`TopK`](crate::TopK) have `Max` and `Min`.
#[cube]
pub trait Carrier: CubeType + Sized {
    /// `lhs ∗ rhs` under `monoid`.
    fn combine(lhs: Self, rhs: Self, #[comptime] monoid: Monoid) -> Self;

    /// The value `monoid` leaves every other value unchanged with.
    fn identity(#[comptime] monoid: Monoid) -> Self;

    /// The partials of one cell, held as `share` says, combined under `monoid`, the total left on
    /// every holder.
    fn across(value: Self, #[comptime] share: UnitShare, #[comptime] monoid: Monoid) -> Self;
}

/// Numbers and their vectors, pointwise. On floats, `Max` and `Min` keep NaN, from `-inf` and
/// `+inf` ([`Monoid`]).
#[cube]
impl<T> Carrier for T
where
    T: CubePrimitive<Scalar: Numeric>
        + Cast
        + CubePartialOrd
        + CubeAdd
        + CubeMul
        + core::ops::Add<T, Output = T>
        + core::ops::Mul<T, Output = T>,
{
    fn combine(lhs: T, rhs: T, #[comptime] monoid: Monoid) -> T {
        let elem = elem_type_of::<T>();
        let float = comptime!(elem.is_float());
        match comptime!(monoid) {
            Monoid::Sum => lhs + rhs,
            Monoid::Prod => lhs * rhs,
            Monoid::Max => {
                if float {
                    max_keeping_nan::<T>(lhs, rhs)
                } else {
                    max(lhs, rhs)
                }
            }
            Monoid::Min => {
                if float {
                    min_keeping_nan::<T>(lhs, rhs)
                } else {
                    min(lhs, rhs)
                }
            }
        }
    }

    fn identity(#[comptime] monoid: Monoid) -> T {
        T::cast_from(Monoid::identity::<T::Scalar>(monoid))
    }

    fn across(value: T, #[comptime] share: UnitShare, #[comptime] monoid: Monoid) -> T {
        match comptime!(share) {
            UnitShare::Repeated | UnitShare::Whole => value,
            UnitShare::Plane => match comptime!(monoid) {
                Monoid::Sum => plane_sum(value),
                Monoid::Prod => plane_prod(value),
                Monoid::Max => plane_keeping_nan::<T>(value, plane_max(value), monoid),
                Monoid::Min => plane_keeping_nan::<T>(value, plane_min(value), monoid),
            },
            UnitShare::Group { unit_bits } => {
                let mut total = value;
                #[unroll]
                for bit in 0..comptime!(usize::BITS - unit_bits.leading_zeros()) {
                    if comptime!(unit_bits & (1 << bit) != 0) {
                        total = monoid
                            .combine::<T>(total, plane_shuffle_xor(total, comptime!(1u32 << bit)));
                    }
                }
                total
            }
        }
    }
}

/// `ordered`, the plane's max or min of `value` as the plane instruction gives it, with a NaN put
/// back where any unit held one: the instruction drops NaN, the monoid keeps it.
///
/// Each unit sends `-inf` under `Max` (`+inf` under `Min`), or NaN where it holds one: its value
/// pushed past that infinity and back, which only a NaN survives. The plane sums them, which keeps
/// NaN, so the sum is that infinity or NaN; and combining `ordered` with it leaves `ordered`,
/// infinities and zeros of either sign included, or gives NaN. Integers hold no NaN.
#[cube]
fn plane_keeping_nan<T: CubePrimitive<Scalar: Numeric> + Cast + Carrier>(
    value: T,
    ordered: T,
    #[comptime] monoid: Monoid,
) -> T {
    let elem = elem_type_of::<T>();
    if comptime!(elem.is_float()) {
        let far = match comptime!(monoid) {
            Monoid::Max => T::cast_from(f32::NEG_INFINITY),
            _ => T::cast_from(f32::INFINITY),
        };
        let nan_or_far = match comptime!(monoid) {
            Monoid::Max => max_keeping_nan::<T>(min_keeping_nan::<T>(value, far), far),
            _ => min_keeping_nan::<T>(max_keeping_nan::<T>(value, far), far),
        };
        monoid.combine::<T>(ordered, plane_sum(nan_or_far))
    } else {
        ordered
    }
}

/// The larger of two float carriers, NaN on either side kept. Any float carrier, so a carrier
/// whose element is known only at expansion can call it; `cubecl`'s own `max_nan` asks the type
/// for a float element.
#[cube]
pub(crate) fn max_keeping_nan<T: CubePrimitive>(lhs: T, rhs: T) -> T {
    intrinsic!(|scope| {
        let (lhs, rhs) = (lhs.read_value(scope), rhs.read_value(scope));
        let op = FMaxNanOp::new(scope.ctx_mut(), lhs, rhs);
        scope.register_with_result(&op).into()
    })
}

/// The smaller of two float carriers, NaN on either side kept ([`max_keeping_nan`]).
#[cube]
pub(crate) fn min_keeping_nan<T: CubePrimitive>(lhs: T, rhs: T) -> T {
    intrinsic!(|scope| {
        let (lhs, rhs) = (lhs.read_value(scope), rhs.read_value(scope));
        let op = FMinNanOp::new(scope.ctx_mut(), lhs, rhs);
        scope.register_with_result(&op).into()
    })
}

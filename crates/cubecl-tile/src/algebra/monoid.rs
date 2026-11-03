//! An identity and an associative operation.

use cubecl::ir::dialect::cmp::{FMaxNanOp, FMinNanOp};
use cubecl::prelude::*;

use crate::comptime_only;

/// The ordering and arithmetic a [`Monoid`] needs of the values it combines.
pub trait Carrier:
    CubePartialOrd
    + CubeAdd
    + CubeMul
    + core::ops::Add<Self, Output = Self>
    + core::ops::Mul<Self, Output = Self>
    + Sized
{
}

impl<T> Carrier for T where
    T: CubePartialOrd
        + CubeAdd
        + CubeMul
        + core::ops::Add<Self, Output = Self>
        + core::ops::Mul<Self, Output = Self>
{
}

/// An identity and an associative operation, used by everything that merges values.
///
/// On floats, `Max` and `Min` keep NaN: a NaN on either side is the result, as `max(NaN, x)` is
/// in a reduction that must not hide one. Their identities are `-inf` and `+inf`, so a run of
/// `-inf` stays `-inf` under `Max`, where the lowest finite value would replace it. On integers
/// they are the plain order, with the type's bounds as identities.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Monoid {
    /// `acc + val`, identity `0`.
    Sum,
    /// `acc * val`, identity `1`.
    Prod,
    /// `max(acc, val)`, NaN kept; identity `-inf` on floats, the lowest value on integers.
    Max,
    /// `min(acc, val)`, NaN kept; identity `+inf` on floats, the highest value on integers.
    Min,
}

#[cube]
impl Monoid {
    /// This monoid's identity element; a masked read past an operand's extent must return it.
    pub fn identity<E: Numeric>(#[comptime] monoid: Monoid) -> E {
        let elem = elem_type_of::<E>();
        let float = comptime!(elem.is_float());
        match comptime!(monoid) {
            Monoid::Sum => E::from_int(0),
            Monoid::Prod => E::from_int(1),
            Monoid::Max => {
                if float {
                    E::cast_from(f32::NEG_INFINITY)
                } else {
                    E::min_value()
                }
            }
            Monoid::Min => {
                if float {
                    E::cast_from(f32::INFINITY)
                } else {
                    E::max_value()
                }
            }
        }
    }

    /// `lhs ∗ rhs`, pointwise on vectors.
    fn combine_of<T: Carrier + CubePrimitive>(lhs: T, rhs: T, #[comptime] monoid: Monoid) -> T {
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

    /// A vector's first `width` components combined into one value.
    pub fn reduce<E: Numeric, N: Size>(
        v: Vector<E, N>,
        #[comptime] width: usize,
        #[comptime] monoid: Monoid,
    ) -> E {
        let mut acc = Monoid::identity::<E>(monoid);
        #[unroll]
        for j in 0..width {
            acc = monoid.combine::<E>(acc, v.extract(j));
        }
        acc
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

/// `monoid.combine(a, b)`, hand-written since a comptime-only type has no `{Name}Expand`.
impl Monoid {
    pub fn combine<T: Carrier + CubePrimitive>(self, lhs: T, rhs: T) -> T {
        Monoid::combine_of::<T>(lhs, rhs, self)
    }

    pub fn __expand_combine_method<T: Carrier + CubePrimitive>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
    ) -> T::ExpandType {
        Monoid::__expand_combine_of::<T>(scope, lhs, rhs, self)
    }
}

comptime_only!(Monoid);

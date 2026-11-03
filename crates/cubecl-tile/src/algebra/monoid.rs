//! An identity and an associative operation.

use cubecl::prelude::*;

use crate::Carrier;
use crate::comptime_only;

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

/// `monoid.combine(a, b)`, the carrier's [`Carrier::combine`]; hand-written since a comptime-only type has no `{Name}Expand`.
impl Monoid {
    pub fn combine<T: Carrier>(self, lhs: T, rhs: T) -> T {
        T::combine(lhs, rhs, self)
    }

    pub fn __expand_combine_method<T: Carrier>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
    ) -> T::ExpandType {
        T::__expand_combine(scope, lhs, rhs, self)
    }
}

comptime_only!(Monoid);

//! An identity and an associative operation.

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Monoid {
    /// `acc + val`, identity `0`.
    Sum,
    /// `acc * val`, identity `1`.
    Prod,
    /// `max(acc, val)`, identity the lowest value of the element.
    Max,
    /// `min(acc, val)`, identity the highest value of the element.
    Min,
}

#[cube]
impl Monoid {
    /// This monoid's identity element; a masked read past an operand's extent must return it.
    pub fn identity<E: Numeric>(#[comptime] monoid: Monoid) -> E {
        match comptime!(monoid) {
            Monoid::Sum => E::from_int(0),
            Monoid::Prod => E::from_int(1),
            Monoid::Max => E::min_value(),
            Monoid::Min => E::max_value(),
        }
    }

    /// `lhs ∗ rhs`, pointwise on vectors.
    fn combine_of<T: Carrier>(lhs: T, rhs: T, #[comptime] monoid: Monoid) -> T {
        match comptime!(monoid) {
            Monoid::Sum => lhs + rhs,
            Monoid::Prod => lhs * rhs,
            Monoid::Max => max(lhs, rhs),
            Monoid::Min => min(lhs, rhs),
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

    /// `seed ∗ arr₀ ∗ … ∗ arr_{len-1}`: `len` elements of `arr` combined into `seed`.
    pub fn reduce_array<E: Numeric>(
        arr: &Array<E>,
        #[comptime] len: usize,
        seed: E,
        #[comptime] monoid: Monoid,
    ) -> E {
        let mut acc = seed;
        #[unroll]
        for i in 0..len {
            acc = monoid.combine::<E>(acc, arr[i]);
        }
        acc
    }
}

/// `monoid.combine(a, b)`, hand-written since a comptime-only type has no `{Name}Expand`.
impl Monoid {
    pub fn combine<T: Carrier>(self, lhs: T, rhs: T) -> T {
        Monoid::combine_of::<T>(lhs, rhs, self)
    }

    pub fn __expand_combine_method<T: Carrier>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
    ) -> T::ExpandType {
        Monoid::__expand_combine_of::<T>(scope, lhs, rhs, self)
    }
}

comptime_only!(Monoid);

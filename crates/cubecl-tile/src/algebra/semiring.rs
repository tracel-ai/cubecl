//! The pair of monoids a contraction runs under.

use cubecl::prelude::*;

use super::monoid::{Carrier, Monoid};
use crate::comptime_only;

/// A product monoid and the accumulation monoid its products fold into.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Semiring {
    add: Monoid,
    mul: Monoid,
}

impl Semiring {
    /// `(+, *)`: the ordinary matmul.
    pub const SUM_PROD: Self = Self {
        add: Monoid::Sum,
        mul: Monoid::Prod,
    };
    /// `(min, +)`: shortest paths, DTW.
    pub const MIN_SUM: Self = Self {
        add: Monoid::Min,
        mul: Monoid::Sum,
    };
    /// `(max, +)`: Viterbi in log space.
    pub const MAX_SUM: Self = Self {
        add: Monoid::Max,
        mul: Monoid::Sum,
    };

    /// The monoid products accumulate into, and every drain folds partials under.
    pub const fn add(self) -> Monoid {
        self.add
    }

    /// The product a pair of operands forms before it is folded in.
    pub const fn mul(self) -> Monoid {
        self.mul
    }
}

#[cube]
impl Semiring {
    /// One accumulation step, `acc + (lhs * rhs)`, in `fma`'s argument order.
    /// [`SUM_PROD`](Semiring::SUM_PROD) must stay a single `fma`.
    fn step_of<T: Carrier + CubePrimitive>(
        lhs: T,
        rhs: T,
        acc: T,
        #[comptime] semiring: Semiring,
    ) -> T {
        if comptime!(semiring == Semiring::SUM_PROD) {
            fma(lhs, rhs, acc)
        } else {
            let product = semiring.mul().combine::<T>(lhs, rhs);
            semiring.add().combine::<T>(product, acc)
        }
    }
}

/// `semiring.step(a, b, acc)`, the pair [`Monoid::combine`] documents.
impl Semiring {
    pub fn step<T: Carrier + CubePrimitive>(self, lhs: T, rhs: T, acc: T) -> T {
        Semiring::step_of::<T>(lhs, rhs, acc, self)
    }

    pub fn __expand_step_method<T: Carrier + CubePrimitive>(
        self,
        scope: &Scope,
        lhs: T::ExpandType,
        rhs: T::ExpandType,
        acc: T::ExpandType,
    ) -> T::ExpandType {
        Semiring::__expand_step_of::<T>(scope, lhs, rhs, acc, self)
    }
}

comptime_only!(Semiring);

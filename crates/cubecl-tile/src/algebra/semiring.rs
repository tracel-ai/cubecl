//! The pair of monoids a contraction runs under.

use cubecl::prelude::*;

use crate::comptime_only;
use crate::{Carrier, Monoid};

/// A product monoid and the accumulation monoid its products sum into.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Semiring {
    /// `(+, *)`: the ordinary matmul.
    SumProd,
    /// `(min, +)`: shortest paths, DTW.
    MinSum,
    /// `(max, +)`: Viterbi in log space.
    MaxSum,
}

impl Semiring {
    /// The monoid products accumulate into, and every drain sums partials under.
    pub const fn add(self) -> Monoid {
        match self {
            Semiring::SumProd => Monoid::Sum,
            Semiring::MinSum => Monoid::Min,
            Semiring::MaxSum => Monoid::Max,
        }
    }

    /// The product a pair of operands forms before it is summed in.
    pub const fn mul(self) -> Monoid {
        match self {
            Semiring::SumProd => Monoid::Prod,
            Semiring::MinSum | Semiring::MaxSum => Monoid::Sum,
        }
    }
}

#[cube]
impl Semiring {
    /// One accumulation step, `acc + (lhs * rhs)`, in `fma`'s argument order.
    /// [`SumProd`](Semiring::SumProd) over floats must stay a single `fma`. An `fma` is a float
    /// instruction, so integers take the product and the sum apart, which is exact.
    fn step_of<T: Carrier + CubePrimitive>(
        lhs: T,
        rhs: T,
        acc: T,
        #[comptime] semiring: Semiring,
    ) -> T {
        let element = elem_type_of::<T>();
        if comptime!(semiring == Semiring::SumProd && element.is_float()) {
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

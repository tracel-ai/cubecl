//! A recipe stated as one factor per contracted axis.

use cubecl::prelude::*;

use super::{FactorReads, Reads, Recipe, RecipeCoords, RecipeExpand, Separable, SeparableExpand};

/// The product of one factor per contracted axis, in contraction order: `K₀ ⊗ K₁ ⊗ … ⊗ Kₙ₋₁`.
/// Each factor must read a distinct axis; this is not checked.
#[derive(CubeType, Clone)]
pub struct Factors<R: CubeType> {
    pub factors: Sequence<R>,
}

impl<R: CubeType> FactorReads for FactorsExpand<R>
where
    R::ExpandType: Reads,
{
    fn factor_reads(&self, scope: &Scope, factor: usize, axis: crate::Axis) -> bool {
        let factor = NativeExpand::from_lit(scope, factor);
        self.factors
            .__expand_index_method(scope, factor)
            .reads(scope, axis)
    }
}

#[cube]
impl<R: CubeType> Factors<R> {
    /// The product of `factors`, in contraction order.
    pub fn new(factors: Sequence<R>) -> Factors<R> {
        Factors::<R> { factors }
    }

    /// The sequence's length; panics when empty.
    pub(crate) fn rank(&self) -> comptime_type!(usize) {
        let rank = self.factors.len();
        comptime!(assert!(
            rank > 0,
            "Factors: a separable recipe needs at least one factor"
        ));
        rank
    }
}

#[cube]
impl<T: Numeric, R: Recipe<T>> Recipe<T> for Factors<R> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let rank = self.rank();
        let mut value = self.factors.index(0usize).evaluate(coordinates);

        #[unroll]
        for f in 1..rank {
            value *= self.factors.index(f).evaluate(coordinates);
        }

        value
    }
}

#[cube]
impl<T: Numeric, R: Recipe<T>> Separable<T> for Factors<R> {
    fn factors(&self) -> comptime_type!(usize) {
        self.rank()
    }

    fn factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T {
        self.factors.index(factor).evaluate(coordinates)
    }
}

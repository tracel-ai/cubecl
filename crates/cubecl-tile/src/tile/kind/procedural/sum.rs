use cubecl::prelude::*;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// Pointwise sum of two recipes: `(A + B)(coords) = A(coords) + B(coords)`.
#[derive(CubeType, Clone)]
pub struct Sum<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

/// Construct a [`Sum`].
#[cube]
pub fn sum_of<A: CubeType, B: CubeType>(lhs: A, rhs: B) -> Sum<A, B> {
    Sum::<A, B> { lhs, rhs }
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Sum<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) + self.rhs.evaluate(coordinates)
    }
}

impl<A: CubeType, B: CubeType> Reads for SumExpand<A, B>
where
    A::ExpandType: Reads,
    B::ExpandType: Reads,
{
    fn reads(&self, scope: &Scope, axis: crate::Axis) -> bool {
        self.lhs.reads(scope, axis) || self.rhs.reads(scope, axis)
    }
}

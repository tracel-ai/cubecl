use cubecl::prelude::*;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// Pointwise product of two recipes: `(A * B)(coords) = A(coords) * B(coords)`.
#[derive(CubeType, Clone)]
pub struct Product<A: CubeType, B: CubeType> {
    pub lhs: A,
    pub rhs: B,
}

/// Construct a [`Product`].
#[cube]
pub fn product_of<A: CubeType, B: CubeType>(lhs: A, rhs: B) -> Product<A, B> {
    Product::<A, B> { lhs, rhs }
}

#[cube]
impl<T: Numeric, A: Recipe<T>, B: Recipe<T>> Recipe<T> for Product<A, B> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.lhs.evaluate(coordinates) * self.rhs.evaluate(coordinates)
    }
}

impl<A: CubeType, B: CubeType> Reads for ProductExpand<A, B>
where
    A::ExpandType: Reads,
    B::ExpandType: Reads,
{
    fn reads(&self, scope: &Scope, axis: crate::Axis) -> bool {
        self.lhs.reads(scope, axis) || self.rhs.reads(scope, axis)
    }
}

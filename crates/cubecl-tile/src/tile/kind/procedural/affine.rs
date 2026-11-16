use cubecl::prelude::*;

use crate::Dimension;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// A one-dimensional affine coordinate expression, `offset + coefficient * coordinate[dimension]`.
#[derive(CubeType, Clone)]
pub struct AffineCoordinate<T: Numeric> {
    pub offset: T,
    pub coefficient: T,
    #[cube(comptime)]
    pub dimension: Dimension,
}

/// Construct an [`AffineCoordinate`] along the specified dimension.
#[cube]
pub fn affine_along<T: Numeric>(
    #[comptime] dimension: Dimension,
    offset: T,
    coefficient: T,
) -> AffineCoordinate<T> {
    AffineCoordinate::<T> {
        offset,
        coefficient,
        dimension,
    }
}

#[cube]
impl<T: Numeric> Recipe<T> for AffineCoordinate<T> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.offset + self.coefficient * T::cast_from(coordinates.along(self.dimension))
    }
}

impl<T: Numeric> Reads for AffineCoordinateExpand<T> {
    fn reads(&self, _scope: &Scope, dimension: Dimension) -> bool {
        self.dimension == dimension
    }
}

use cubecl::prelude::*;

use crate::Axis;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// A one-dimensional affine coordinate expression, `offset + coefficient * coordinate[axis]`.
#[derive(CubeType, Clone)]
pub struct AffineCoordinate<T: Numeric> {
    pub offset: T,
    pub coefficient: T,
    #[cube(comptime)]
    pub axis: Axis,
}

/// Construct an [`AffineCoordinate`] along the specified axis.
#[cube]
pub fn affine_along<T: Numeric>(
    #[comptime] axis: Axis,
    offset: T,
    coefficient: T,
) -> AffineCoordinate<T> {
    AffineCoordinate::<T> {
        offset,
        coefficient,
        axis,
    }
}

#[cube]
impl<T: Numeric> Recipe<T> for AffineCoordinate<T> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        self.offset + self.coefficient * T::cast_from(coordinates.along(self.axis))
    }
}

impl<T: Numeric> Reads for AffineCoordinateExpand<T> {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        self.axis == axis
    }
}

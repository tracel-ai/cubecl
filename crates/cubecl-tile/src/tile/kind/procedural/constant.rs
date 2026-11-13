use cubecl::prelude::*;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// A procedural field holding one value, which may be a runtime scalar.
#[derive(CubeType, Clone)]
pub struct Constant<T: Numeric> {
    pub value: T,
}

#[cube]
impl<T: Numeric> Recipe<T> for Constant<T> {
    fn evaluate(&self, _coordinates: &RecipeCoords) -> T {
        self.value
    }
}

impl<T: Numeric> Reads for ConstantExpand<T> {
    fn reads(&self, _scope: &Scope, _axis: crate::Axis) -> bool {
        false
    }
}

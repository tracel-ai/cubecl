use cubecl::prelude::*;

use crate::{Axis, Integer, IntegerExpand, floor_div_rem};

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// The fractional part a rational coordinate mapping leaves behind, scaled:
/// `coefficient * frac((coord[axis] * numerator_scale + numerator_offset) / divisor)`.
#[derive(CubeType, Clone)]
pub struct Phase<T: Float> {
    /// Multiplies the whole fraction.
    pub coefficient: T,
    pub numerator_scale: u32,
    pub numerator_offset: i32,
    pub divisor: u32,
    #[cube(comptime)]
    pub axis: Axis,
}

#[cube]
impl<T: Float> Recipe<T> for Phase<T> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let divisor = self.divisor.constant();
        // Checked here: a struct literal bypasses any constructor.
        comptime!(assert!(
            divisor != Some(0),
            "Phase: divisor must be non-zero"
        ));
        let numerator = coordinates
            .along(self.axis)
            .times(self.numerator_scale)
            .cast::<i32>()
            .plus(self.numerator_offset);
        let (_, residue) = floor_div_rem(numerator, self.divisor.cast::<i32>());
        let residue = T::cast_from(residue);
        // A folded divisor becomes a reciprocal literal to multiply by.
        let fraction = if comptime!(divisor.is_some()) {
            residue * T::new(comptime!(1.0 / divisor.unwrap() as f64))
        } else {
            residue / T::cast_from(self.divisor)
        };
        self.coefficient * fraction
    }
}

impl<T: Float> Reads for PhaseExpand<T> {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        self.axis == axis
    }
}

//! Recipes: memory-free sources evaluated at logical coordinates, and their factorization.

use cubecl::ir::Scope;
use cubecl::prelude::*;

use crate::{Coords, Dimension, Space};

/// Coordinate dependence of a recipe, queried while a kernel is expanded.
pub trait Reads {
    fn reads(&self, scope: &Scope, dimension: Dimension) -> bool;
}

/// Per-factor coordinate dependencies of a separable recipe, queried during expansion.
pub trait FactorReads {
    fn factor_reads(&self, scope: &Scope, factor: usize, dimension: Dimension) -> bool;
}

/// The absolute logical coordinates a [`Recipe`] is evaluated at: `origin` plus read offset.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RecipeCoords {
    origin: Coords<u32>,
    offset: Coords<u32>,
    #[cube(comptime)]
    pub space: Space,
}

#[cube]
impl RecipeCoords {
    pub(crate) fn new(
        origin: &Coords<u32>,
        offset: &Coords<u32>,
        #[comptime] space: Space,
    ) -> Self {
        RecipeCoords {
            origin: origin.clone(),
            offset: offset.clone(),
            space,
        }
    }

    /// The absolute coordinate along the dimension at comptime position `p`.
    pub fn at(&self, #[comptime] p: usize) -> u32 {
        self.origin.at(p) + self.offset.at(p)
    }

    /// The absolute coordinate along the specified `dimension`.
    pub fn along(&self, #[comptime] dimension: Dimension) -> u32 {
        self.at(comptime!(self.space.position(dimension)))
    }
}

/// An N-dimensional scalar field evaluated at absolute logical coordinates.
#[cube(expand_base_traits = "ExpandTypeClone")]
pub trait Recipe<T: Numeric> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T;
}

/// A [`Recipe`] that factorizes into one factor per contracted dimension, `R = ∏ᵢ Rᵢ`.
/// Factor `i` must vary only along the `i`-th contracted dimension.
#[cube]
pub trait Separable<T: Numeric>: Recipe<T> {
    fn factors(&self) -> comptime_type!(usize);
    /// The factor at comptime position `factor`, in the contraction's walk order of dimensions.
    fn factor(&self, coordinates: &RecipeCoords, #[comptime] factor: usize) -> T;
}

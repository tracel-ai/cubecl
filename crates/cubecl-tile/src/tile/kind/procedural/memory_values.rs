//! A memory tile's values read one at a time, at coordinates: what a recipe built over another
//! operand reads of it.

use cubecl::ir::{ElemType, Scope};
use cubecl::prelude::*;

use super::{Reads, Recipe, RecipeCoords};
use crate::*;

/// A memory tile's values, read one at a time at the coordinates they are asked at. A dimension a
/// read does not name is read where the tile's window sits on it, so a tile cut to one owner's
/// batch and head reads that batch and head; a coordinate past the buffer reads zero. A recipe
/// itself, evaluated at its own coordinates, so it composes with any other
/// ([`Sum`](crate::procedural::Sum)).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemoryValues {
    memory: Memory,
    #[cube(comptime)]
    element: ElemType,
}

#[cube]
impl Tile {
    /// This tile's values, read one at a time ([`MemoryValues`]).
    ///
    /// # Panics
    ///
    /// At expansion, on a tile that is not memory.
    pub fn values(&self) -> MemoryValues {
        match &self.kind {
            TileKind::Memory(memory) => MemoryValues {
                memory: memory.clone(),
                element: comptime!(self.element),
            },
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::values: only a memory tile is read one value at a time")
            }
        }
    }
}

#[cube]
impl MemoryValues {
    /// The value at `coordinates`, one along each of `dimensions` in order, as `T`.
    pub fn read<T: Numeric>(
        &self,
        #[comptime] dimensions: Vec<Dimension>,
        coordinates: &Sequence<u32>,
    ) -> T {
        let mut at = Coords::<u32>::new();
        #[unroll]
        for p in 0..dimensions.len() {
            at.push(*coordinates.index(p));
        }
        self.read_at::<T>(dimensions, &at)
    }

    /// [`read`](Self::read) at coordinates that keep their constness.
    pub(crate) fn read_at<T: Numeric>(
        &self,
        #[comptime] dimensions: Vec<Dimension>,
        coordinates: &Coords<u32>,
    ) -> T {
        let define!(Stored) = self.element;
        let logical = comptime!(self.memory.projection.dimensions().to_vec());
        let mut position = Coords::<u32>::new();
        #[allow(
            clippy::needless_range_loop,
            reason = "an unrolled loop over a comptime range; `p` also reads the window's origin"
        )]
        #[unroll]
        for p in 0..logical.len() {
            let dimension = comptime!(logical[p]);
            match comptime!(dimensions.iter().position(|&named| named == dimension)) {
                Some(named) => position.push(coordinates.at(named)),
                None => position.push(u32::cast_from(self.memory.window.origin.at(p))),
            }
        }
        let size!(One) = 1usize;
        let view = self.memory.coordinate_view::<Stored, One>();
        T::cast_from(view.read_checked(position.to_dyn()).extract(0usize))
    }
}

#[cube]
impl<T: Numeric> Recipe<T> for MemoryValues {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let dimensions = comptime!(coordinates.space.dimensions().collect::<Vec<_>>());
        let mut at = Coords::<u32>::new();
        #[unroll]
        for p in 0..dimensions.len() {
            at.push(coordinates.at(p));
        }
        self.read_at::<T>(dimensions, &at)
    }
}

impl Reads for MemoryValuesExpand {
    fn reads(&self, _scope: &Scope, dimension: Dimension) -> bool {
        self.memory.projection.dimensions().contains(&dimension)
    }
}

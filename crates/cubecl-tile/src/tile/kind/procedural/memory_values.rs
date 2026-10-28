//! A memory tile's values read one at a time, at coordinates: what a recipe built over another
//! operand reads of it.

use cubecl::ir::{ElemType, Scope};
use cubecl::prelude::*;

use super::{Reads, Recipe, RecipeCoords};
use crate::*;

/// A memory tile's values, read one at a time at the coordinates they are asked at. An axis a
/// read does not name is read where the tile's window sits on it, so a tile cut to one owner's
/// batch and head reads that batch and head; a coordinate past the buffer reads zero. A recipe
/// itself, evaluated at its own coordinates, so it composes with any other ([`Sum`](super::Sum)).
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
    /// The value at `coordinates`, one along each of `axes` in order, as `T`.
    pub fn read<T: Numeric>(&self, #[comptime] axes: Vec<Axis>, coordinates: &Coords<u32>) -> T {
        let define!(Stored) = self.element;
        let logical = comptime!(self.memory.projection.logical_axes().to_vec());
        let mut position = Coords::<u32>::new();
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for p in 0..logical.len() {
            let axis = comptime!(logical[p]);
            match comptime!(axes.iter().position(|&named| named == axis)) {
                Some(named) => position.push(coordinates.at(named)),
                None => position.push(u32::cast_from(self.memory.window.origin.at(p))),
            }
        }
        let size!(One) = 1usize;
        let view = self.memory.read_view::<Stored, One>(self.memory.base());
        T::cast_from(view.read_checked(position.to_dyn()).extract(0usize))
    }
}

#[cube]
impl<T: Numeric> Recipe<T> for MemoryValues {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let axes = comptime!(coordinates.space.axes().collect::<Vec<_>>());
        let mut at = Coords::<u32>::new();
        #[unroll]
        for p in 0..axes.len() {
            at.push(coordinates.at(p));
        }
        self.read::<T>(axes, &at)
    }
}

impl Reads for MemoryValuesExpand {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        self.memory.projection.logical_axes().contains(&axis)
    }
}

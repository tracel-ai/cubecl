//! A cube's output as its planes reach it: written by each plane where every plane owns its box of
//! it, or summed across the planes that share a box before the cube writes it.

use cubecl::prelude::*;

use crate::tile::accumulator::smem_cyclic::SmemCyclicAccumulation;
use crate::*;

/// The output of one cube, reached by its planes.
///
/// Where the partitioning hands every plane a box of the output of its own, a plane's sum drains
/// into its box. Where it splits an axis the output lacks across the planes, the contracted axis of
/// a sum they each hold part of, the planes holding the same box drain into a shared-memory
/// accumulator on the cyclic schedule ([`SmemCyclicAccumulation`]) and [`write`] stores the sum.
/// The tile says which from the partitioning it was cut by: the caller states nothing.
///
/// [`write`]: PlanesOutput::write
#[derive(CubeType)]
pub struct PlanesOutput<A: Numeric> {
    out: Tile,
    merged: ComptimeOption<SmemCyclicAccumulation<A>>,
    /// The axes the output lacks that the planes are split along, each with its count of planes.
    #[cube(comptime)]
    sharing: Vec<(Axis, usize)>,
}

#[cube]
impl Tile {
    /// This tile, `cube`'s window of an output, as its planes reach it, a sum shared by planes
    /// added in `A`.
    pub fn planes_output<A: Numeric>(&self, cube: &Region) -> PlanesOutput<A> {
        let space = comptime!(self.place.space.clone());
        let sharing = comptime!(cube.path.planes_along(|axis| !space.contains(axis)));
        let writers = comptime!(sharing.iter().map(|&(_, count)| count).product::<usize>());
        let merged = if comptime!(writers == 1) {
            ComptimeOption::new_None()
        } else {
            ComptimeOption::new_Some(self.smem_cyclic_accumulation::<A>(writers))
        };
        PlanesOutput::<A> {
            out: self.clone(),
            merged,
            sharing,
        }
    }
}

#[cube]
impl<A: Numeric> PlanesOutput<A> {
    /// Drain `partial`, the sum `plane` holds over its box of the output. Where planes share the
    /// box it is a meeting of the cube: every unit of every plane drains, each with its plane's
    /// partial.
    pub fn drain(&self, partial: &Tile, plane: &Region) {
        #[comptime]
        match &self.merged {
            ComptimeOption::None => partial.drained_into(&self.out.at(plane)),
            ComptimeOption::Some(merged) => {
                merged.drain_at(partial, plane, PlanesOutput::<A>::writer(self, plane))
            }
        }
    }

    /// Store the sum the planes sharing a box drained into it, once every plane has; nothing where
    /// each plane drained into its own box.
    pub fn write(&mut self) {
        #[comptime]
        match &self.merged {
            ComptimeOption::None => {}
            ComptimeOption::Some(merged) => self.out.copy_from(&merged.source),
        }
    }

    /// `plane`'s place among the planes sharing its box.
    fn writer(&self, plane: &Region) -> usize {
        let mut writer = 0usize;
        #[unroll]
        for i in 0..comptime!(self.sharing.len()) {
            let (axis, count) = comptime!(self.sharing[i]);
            writer = writer * count + plane.coord(axis);
        }
        writer
    }
}

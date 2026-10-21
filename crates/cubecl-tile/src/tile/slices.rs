//! A tile cut into slices along one axis, where its holder keeps them ([`AxisSlices`]).

use cubecl::prelude::*;

use crate::*;

/// Who keeps a tile's slices, and so how a slice is read whole.
#[allow(clippy::large_enum_variant)]
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum AxisSlicesKind {
    /// A register block a unit holds whole: a slice runs along its columns, in its registers.
    Registers(RegisterData),
    /// A window of shared memory a plane holds: its units take a slice's cells in turns and meet
    /// on the slice's partials.
    Window(Memory),
    /// A plane's grid of fragments. A manual-mma fragment's cells are read in its registers, each
    /// unit keeping the rows it holds cells of; a cmma fragment's lie where the instruction keeps
    /// them, and are only scaled, through the plane's scratch.
    Fragments(PlanePartition),
    /// One fragment of a plane's, reached the same way.
    Fragment(PlaneTile),
}

/// A tile cut into slices along `axis`, one for every index of its other axes, each read and
/// written whole where the tile's holder keeps it: the verbs an op reducing or broadcasting along
/// one axis runs. The slices write through to the tile.
///
/// `axis` is the tile's innermost, the one its cells are contiguous along; any other is refused
/// by name.
#[derive(CubeType)]
pub struct AxisSlices {
    kind: AxisSlicesKind,
    #[cube(comptime)]
    space: Space,
    #[cube(comptime)]
    axis: Axis,
}

#[cube]
impl Tile {
    /// This tile cut into slices along `axis`, where its holder keeps them: a unit's register
    /// block, a plane's window of shared memory, or a plane's fragments. A window of memory any
    /// wider holder shares has no one to read a slice whole, and is refused.
    pub fn along(&self, #[comptime] axis: Axis) -> AxisSlices {
        let holder = comptime!(self.place.holder());
        let space = comptime!(self.place.space.clone());
        comptime!(assert!(
            space.axis_at(space.rank() - 1) == axis,
            "Tile::along: a slice runs along the tile's innermost axis, and {axis:?} is not the \
             innermost of {space:?}"
        ));
        let kind = match &self.kind {
            TileKind::PlanePartition(partition) => partition.slices(),
            TileKind::PlaneTile(tile) => tile.slices(),
            TileKind::Memory(window) => {
                comptime!(assert!(
                    holder == ComputeScope::Plane,
                    "Tile::along: a window of shared memory is read slice by slice by the plane \
                     that holds it; this one is held by {holder:?}"
                ));
                AxisSlicesKind::new_Window(window.clone())
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => panic!(
                "Tile::along: a register block, a plane's fragments, or a plane's window holds \
                 slices"
            ),
        };
        AxisSlices { kind, space, axis }
    }
}

#[cube]
impl Tile {
    /// Slices along `axis` one unit keeps a state for: the rows a manual-mma grid's unit holds
    /// cells of, which a reduction reads in its registers; every slice of any other holder.
    pub fn slices_per_unit(&self, #[comptime] axis: Axis) -> comptime_type!(usize) {
        let define!(Served) = self.element;
        let every = comptime!(self.place.space.slices_along(axis));
        match &self.kind {
            TileKind::PlanePartition(partition) => match partition.at(0usize, 0usize) {
                PlaneTile::Mma(fragment) => {
                    let rows = fragment.rows_per_unit::<Served>();
                    comptime!(partition.m_tiles * rows)
                }
                PlaneTile::Cmma(_) | PlaneTile::Registers(_) => comptime!(every),
                PlaneTile::Wgmma(_) => panic!(
                    "Tile::slices_per_unit: a warpgroup accumulator's slices are not read in \
                     place; drain it first"
                ),
            },
            TileKind::PlaneTile(tile) => match tile {
                PlaneTile::Mma(fragment) => fragment.rows_per_unit::<Served>(),
                PlaneTile::Cmma(_) | PlaneTile::Registers(_) => comptime!(every),
                PlaneTile::Wgmma(_) => panic!(
                    "Tile::slices_per_unit: a warpgroup accumulator's slices are not read in \
                     place; drain it first"
                ),
            },
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(every),
        }
    }
}

#[cube]
impl AxisSlices {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]`, the procedural `bias` read at each cell's
    /// coordinates, windowed to the same region as the tile.
    pub(crate) fn scale_add<E: Float>(&mut self, scale: E, bias: &Tile) {
        let recipe = bias.recipe();
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => {
                block.scale_add_along::<E>(scale, recipe, self.space.clone())
            }
            AxisSlicesKind::Window(window) => {
                window.scale_add_along::<E>(scale, recipe, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(partition) => {
                partition.scale_add_along::<E>(scale, recipe, self.space.clone())
            }
            AxisSlicesKind::Fragment(tile) => {
                let mut fragment = tile.manual_fragment();
                fragment.scale_add_along::<E>(scale, recipe, self.space.clone(), (0usize, 0usize))
            }
        }
    }

    /// Each slice's max, starting from `seed`'s.
    pub(crate) fn maxima<E: Float>(&self, seed: &Array<E>) -> Array<E> {
        match &self.kind {
            AxisSlicesKind::Registers(block) => block.maxima_along::<E>(seed),
            AxisSlicesKind::Window(window) => {
                window.maxima_along::<E>(seed, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(partition) => partition.maxima_along::<E>(seed),
            AxisSlicesKind::Fragment(tile) => tile.manual_fragment().maxima_along(seed),
        }
    }

    /// `self[i, s] = exp(self[i, s] − slices[i])` ([`exp_minus_cell`](Self::exp_minus_cell)).
    pub(crate) fn exp_minus<E: Float>(&mut self, slices: &Array<E>) {
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => block.exp_minus_along::<E>(slices),
            AxisSlicesKind::Window(window) => {
                window.exp_minus_along::<E>(slices, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(partition) => partition.exp_minus_along::<E>(slices),
            AxisSlicesKind::Fragment(tile) => {
                let mut fragment = tile.manual_fragment();
                fragment.exp_minus_along(slices, 0usize)
            }
        }
    }

    /// Each slice's sum.
    pub(crate) fn sums<E: Float>(&self) -> Array<E> {
        match &self.kind {
            AxisSlicesKind::Registers(block) => block.sums_along::<E>(),
            AxisSlicesKind::Window(window) => window.sums_along::<E>(self.space.clone(), self.axis),
            AxisSlicesKind::Fragments(partition) => partition.sums_along::<E>(),
            AxisSlicesKind::Fragment(tile) => tile.manual_fragment().sums_along(),
        }
    }

    /// `self[i, s] *= factors[i]`. A plane's fragments bounce through its scratch, met on
    /// `sync_plane`: every unit of the plane calls it.
    pub fn mul<E: Float>(&mut self, factors: &Array<E>) {
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => block.mul_along::<E>(factors, 0usize),
            AxisSlicesKind::Window(window) => {
                window.mul_along::<E>(factors, self.space.clone(), self.axis)
            }
            AxisSlicesKind::Fragments(partition) => partition.mul_along::<E>(factors),
            AxisSlicesKind::Fragment(tile) => tile.mul_along(factors, 0usize),
        }
    }

    /// `values`, the slots a unit of a manual-mma grid keeps for the rows it holds cells of
    /// ([`Tile::slices_per_unit`]), written into `window`, a plane's window of one cell a slice in
    /// slice order: each slice written once, by a unit holding it. The caller meets the plane
    /// before the window is read.
    pub(crate) fn store_held<E: Float>(&self, values: &Array<E>, window: &mut Tile) {
        comptime!(assert!(
            window.place.holder() == ComputeScope::Plane
                && window.place.space.cells() == self.space.slices_along(self.axis),
            "AxisSlices::store_held: a plane's window of one cell for each of the {} slices of \
             {:?}, not {:?} held by {:?}",
            self.space.slices_along(self.axis),
            self.space,
            window.place.space,
            window.place.holder()
        ));
        match &mut window.kind {
            TileKind::Memory(memory) => self.store_held_in::<E>(values, memory),
            TileKind::PlanePartition(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "AxisSlices::store_held: the slices' state lands in a window of shared memory"
            ),
        }
    }

    /// [`store_held`](Self::store_held) into the window's memory.
    fn store_held_in<E: Float>(&self, values: &Array<E>, memory: &mut Memory) {
        match &self.kind {
            AxisSlicesKind::Fragments(partition) => partition.store_along::<E>(values, memory),
            AxisSlicesKind::Fragment(tile) => tile
                .manual_fragment()
                .store_along::<E>(values, memory, 0usize, 0usize),
            AxisSlicesKind::Registers(_) | AxisSlicesKind::Window(_) => panic!(
                "AxisSlices::store_held: a unit's block and a plane's window keep every slice's \
                 slot alike; only a manual-mma grid's units keep their own rows"
            ),
        }
    }

    /// `exp(value − slice)`, zero where `slice` is itself masked: a slice with nothing live yet,
    /// whose max is still the minimum value.
    pub(crate) fn exp_minus_cell<E: Float>(value: E, slice: E) -> E {
        let live = slice > E::min_value();
        select(live, (value - slice).exp(), E::from_int(0))
    }
}

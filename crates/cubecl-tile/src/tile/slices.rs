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

    /// `self[c] = exp(self[c] · scale + bias[c] − logsumexp[c])`: the probabilities of a softmax
    /// whose logsumexp is known, read back from it instead of walking the score again.
    /// `logsumexp` is a window of memory that omits the axis the softmax ran along, which it
    /// broadcasts over: along the slices where that axis is the innermost, along each slice where
    /// it is not, as a walk owned by the keys holds its scores keys first. Its other axes are this
    /// tile's, or one wide. A cell whose logsumexp is the minimum value, every key of its softmax
    /// masked, is left at zero. A plane's manual-mma fragments read each cell's logsumexp in
    /// place, a row's or a column's, so a walk owned by the keys keeps its scores in registers.
    pub fn probabilities<E: Float>(&mut self, scale: E, bias: &Tile, logsumexp: &Tile) {
        let recipe = bias.recipe();
        let broadcast = comptime!(Broadcast::new(&self.space, &logsumexp.place.space));
        let count = comptime!(logsumexp.place.space.cells());
        match &mut self.kind {
            AxisSlicesKind::Registers(block) => block.probabilities_along::<E>(
                scale,
                recipe,
                &logsumexp.cells::<E>(count),
                self.space.clone(),
                broadcast,
            ),
            AxisSlicesKind::Window(window) => window.probabilities_along::<E>(
                scale,
                recipe,
                &logsumexp.cells::<E>(count),
                self.space.clone(),
                broadcast,
            ),
            AxisSlicesKind::Fragments(partition) => partition.probabilities_along::<E>(
                scale,
                recipe,
                logsumexp.mem("AxisSlices::probabilities"),
                self.space.clone(),
                broadcast,
            ),
            AxisSlicesKind::Fragment(tile) => {
                let mut fragment = tile.manual_fragment();
                fragment.probabilities_along::<E>(
                    scale,
                    recipe,
                    logsumexp.mem("AxisSlices::probabilities"),
                    self.space.clone(),
                    broadcast,
                    (0usize, 0usize),
                )
            }
        }
    }

    /// `self[c] = probabilities[c] · (self[c] − delta[c])`: a softmax's gradient against its
    /// scores, where `self` holds the gradient against its probabilities and `delta` is, for each
    /// softmax, `Σ probabilities · self` along the axis it ran along. `delta` broadcasts as
    /// [`probabilities`](Self::probabilities)'s logsumexp does; the probabilities are cut as
    /// `self` is, held by the same holder: a unit's block, a plane's window, or a plane's
    /// manual-mma fragments laid alike.
    pub fn softmax_gradient<E: Float>(&mut self, probabilities: &AxisSlices, delta: &Tile) {
        comptime!(assert!(
            self.space == probabilities.space && self.axis == probabilities.axis,
            "AxisSlices::softmax_gradient: the probabilities are cut as the gradient is, along \
             {:?} of {:?}, not along {:?} of {:?}",
            self.axis,
            self.space,
            probabilities.axis,
            probabilities.space
        ));
        let broadcast = comptime!(Broadcast::new(&self.space, &delta.place.space));
        let count = comptime!(delta.place.space.cells());
        match (&mut self.kind, &probabilities.kind) {
            (AxisSlicesKind::Registers(block), AxisSlicesKind::Registers(probabilities)) => block
                .softmax_gradient_along::<E>(probabilities, &delta.cells::<E>(count), broadcast),
            (AxisSlicesKind::Window(window), AxisSlicesKind::Window(probabilities)) => window
                .softmax_gradient_along::<E>(
                    probabilities,
                    &delta.cells::<E>(count),
                    self.space.clone(),
                    broadcast,
                ),
            (AxisSlicesKind::Fragments(partition), AxisSlicesKind::Fragments(probabilities)) => {
                partition.softmax_gradient_along::<E>(
                    probabilities,
                    delta.mem("AxisSlices::softmax_gradient"),
                    broadcast,
                )
            }
            (AxisSlicesKind::Fragment(tile), AxisSlicesKind::Fragment(probabilities)) => {
                let mut fragment = tile.manual_fragment();
                fragment.softmax_gradient_along::<E>(
                    &probabilities.manual_fragment(),
                    delta.mem("AxisSlices::softmax_gradient"),
                    broadcast,
                    (0usize, 0usize),
                )
            }
            _ => panic!(
                "AxisSlices::softmax_gradient: the gradient and its probabilities are held alike, \
                 both in a unit's block, a plane's window or a plane's fragments"
            ),
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

/// Where each cell of a tile cut into slices along its innermost axis reads a tile that omits
/// some of its axes, which it broadcasts over: the read cell is an offset for each slice plus one
/// step for each position along the slice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Broadcast {
    /// The cell of the broadcast tile each slice's first position reads.
    pub(crate) slice: Vec<usize>,
    /// Cells of the broadcast tile one position along a slice moves: zero where the slices run
    /// along an omitted axis.
    pub(crate) position: usize,
}

impl Broadcast {
    /// How a tile over `space` reads one over `broadcast`, whose every axis is one of `space`'s
    /// at its extent, or one wide.
    ///
    /// # Panics
    ///
    /// Where an axis of `broadcast` is neither.
    pub(crate) fn new(space: &Space, broadcast: &Space) -> Self {
        assert!(
            broadcast.axes().all(|axis| match space.contains(axis) {
                true => space.extent(axis) == broadcast.extent(axis),
                false => broadcast.extent(axis) == 1,
            }),
            "Broadcast::new: every axis of {broadcast:?} is one of {space:?}'s at its extent, or \
             one wide"
        );
        let stride = |axis: Axis| {
            (broadcast.position(axis) + 1..broadcast.rank())
                .map(|p| broadcast.extent_at(p))
                .product::<usize>()
        };
        let innermost = space.axis_at(space.rank() - 1);
        let outer: Vec<Axis> = (0..space.rank() - 1).map(|p| space.axis_at(p)).collect();
        let slices = outer.iter().map(|&a| space.extent(a)).product::<usize>();
        let slice = (0..slices)
            .map(|index| {
                let mut rest = index;
                let mut offset = 0;
                for &axis in outer.iter().rev() {
                    let extent = space.extent(axis);
                    if broadcast.contains(axis) {
                        offset += (rest % extent) * stride(axis);
                    }
                    rest /= extent;
                }
                offset
            })
            .collect();
        let position = match broadcast.contains(innermost) {
            true => stride(innermost),
            false => 0,
        };
        Self { slice, position }
    }

    /// The broadcast tile's cell that position `position` of slice `slice` reads.
    pub(crate) fn at(&self, slice: usize, position: usize) -> usize {
        self.slice[slice] + position * self.position
    }

    /// The cell the first slice's first position reads and the cells one slice and one position
    /// move it, for slices that read the broadcast tile evenly, as a tile of two axes does: what a
    /// fragment computes each cell's read from, its row the slice and its column the position.
    ///
    /// # Panics
    ///
    /// Where the slices do not step evenly.
    pub(crate) fn strides(&self) -> (usize, usize, usize) {
        let first = self.slice[0];
        let per_slice = self.slice.get(1).map_or(0, |second| second - first);
        assert!(
            self.slice
                .iter()
                .enumerate()
                .all(|(i, &cell)| cell == first + i * per_slice),
            "Broadcast::strides: the slices read {:?}, not evenly",
            self.slice
        );
        (first, per_slice, self.position)
    }
}

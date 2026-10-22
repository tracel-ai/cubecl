//! The gathered N-D read view over a [`Tile`](crate::Tile): [`ProjectionInKernel`] maps a logical
//! coordinate to the physical one its window is boxed in, [`CompactionStep`] physical to physical.

use cubecl::prelude::*;

use cubecl_std::tensor::{
    View,
    layout::{Coordinates, CoordsDyn, Layout, LayoutExpand},
};

use crate::*;

/// Any [`Layout`] from a coordinate onto the window's `CoordsDyn`, cloneable in both worlds.
pub(crate) trait LogicalLayout:
    Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

impl<L> LogicalLayout for L where
    L: Layout<SourceCoordinates = CoordsDyn> + Clone + 'static + CubeType<ExpandType: Clone>
{
}

/// [`LogicalLayout`] answering a particular coordinate `C`, for the readers that name one.
pub(crate) trait TileLayout<C: Coordinates>:
    LogicalLayout + Layout<Coordinates = C>
{
}

impl<C: Coordinates, L> TileLayout<C> for L where L: LogicalLayout + Layout<Coordinates = C> {}

/// Any [`LogicalLayout`] with an operand's [`Projection`] applied under it.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Projected<L: LogicalLayout> {
    inner: L,
    projection: ProjectionInKernel,
}

#[cube]
impl<L: LogicalLayout> Projected<L> {
    pub(crate) fn new(inner: L, projection: ProjectionInKernel) -> Self {
        Projected::<L> { inner, projection }
    }
}

#[cube]
impl<L: LogicalLayout> Layout for Projected<L> {
    type Coordinates = L::Coordinates;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.projection.to_source_pos(self.inner.to_source_pos(pos))
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (inner_pos, inner_in_bounds) = self.inner.to_source_pos_checked(pos);
        let (proj_pos, proj_in_bounds) = self.projection.to_source_pos_checked(inner_pos);
        (proj_pos, inner_in_bounds && proj_in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.inner.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (inner_pos, inner_in_bounds) = self.inner.to_source_pos_checked(pos);
        inner_in_bounds && self.projection.is_in_bounds(inner_pos)
    }
}

#[cube]
impl Tile {
    /// The whole logical box as a writable N-D view; refused where a write would alias.
    pub(crate) fn nd_mut<T: Numeric, W: Size>(&mut self) -> MaskedMut<'_, Vector<T, W>, CoordsDyn> {
        let space = comptime!(self.place.space.clone());
        let g = self.mem_mut("nd_mut");
        comptime!(assert!(
            g.projection.composition() != Composition::Overlapping,
            "Tile::nd_mut: an overlapping operand aliases under a write"
        ));
        let layout = g.axis_projection(space);
        g.masked_mut::<T, W, CoordsDyn, ProjectionInKernel>(layout)
    }

    /// The whole logical box, unpacked if packed, under the guard the reader states.
    /// Claim [`Guard::Proved`] only where [`guard_provable`](Tile::guard_provable) allows.
    pub(crate) fn nd_packed<T: Numeric, W: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<T, W>, CoordsDyn> {
        match &self.kind {
            TileKind::Memory(g) => {
                let layout = g.axis_projection(comptime!(self.place.space.clone()));
                g.packed::<T, W, CoordsDyn, ProjectionInKernel>(layout, guard)
            }
            TileKind::Procedural(data) => {
                procedural_nd::<T, W>(data, comptime!(self.place.space.clone()), guard)
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
            TileKind::Lines(_) => {
                panic!("Tile::nd: the plane's units are read at a coordinate (`scale_at`)")
            }
        }
    }

    /// The raw words a packed tile holds, one coordinate per axis of its [`Space`](crate::Space).
    pub(crate) fn nd_words<WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<u32, WP>, CoordsDyn> {
        let g = self.mem("nd_words");
        let layout = g.axis_projection(comptime!(self.place.space.clone()));
        g.nd_words::<WP>(layout, guard)
    }

    /// Whether [`Guard::Proved`] is sound here: false on a [`Boundary::Clamp`] axis.
    pub(crate) fn guard_provable(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(data) => {
                comptime!(!data.window.boundaries.contains(&Some(Boundary::Clamp)))
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(true),
        }
    }
}

/// A procedural tile's whole box as an N-D view, evaluated at the coordinate.
#[cube]
fn procedural_nd<T: Numeric, W: Size>(
    data: &Procedural,
    #[comptime] space: Space,
    #[comptime] guard: Guard,
) -> Masked<'_, Vector<T, W>, CoordsDyn> {
    let layout = axis_projection(
        comptime!(space.clone()),
        comptime!(Projection::direct_over(&space)),
        RuntimeMap::integral(comptime!(space.rank())),
        comptime!(1usize),
    );
    Masked::new(
        View::<Vector<T, W>, CoordsDyn>::new::<&Procedural, CoordsDyn>(data, layout),
        comptime!(guard.checks() && data.bounds_check),
    )
}

/// A gathered operand split into the map folded once per run and the physical view it addresses.
#[derive(CubeType)]
pub(crate) struct Gathered<'a, T: Numeric, W: Size> {
    pub map: ProjectionInKernel,
    pub view: Masked<'a, Vector<T, W>, CoordsDyn>,
    #[cube(comptime)]
    pub rank: usize,
}

#[cube]
impl<'a, T: Numeric, W: Size> Gathered<'a, T, W> {
    fn new(
        map: ProjectionInKernel,
        view: Masked<'a, Vector<T, W>, CoordsDyn>,
        #[comptime] rank: usize,
    ) -> Self {
        Gathered::<'a, T, W> { map, view, rank }
    }
}

#[cube]
impl Tile {
    /// The map, physical read surface and physical rank to step a gathered operand by hand.
    pub(crate) fn nd_split<T: Numeric, W: Size>(&self) -> Gathered<'_, T, W> {
        let space = comptime!(self.place.space.clone());
        match &self.kind {
            TileKind::Memory(g) => Gathered::new(
                g.axis_projection(comptime!(space.clone())),
                // A folded caller derived its coordinates itself, so both guards stay on.
                g.packed::<T, W, CoordsDyn, CompactionStep>(
                    g.physical_box(),
                    comptime!(Guard::Checked),
                ),
                comptime!(g.projection.physical_rank()),
            ),
            TileKind::Procedural(_) | TileKind::Lines(_) => Gathered::new(
                axis_projection(
                    comptime!(space.clone()),
                    comptime!(Projection::direct_over(&space)),
                    RuntimeMap::integral(comptime!(space.rank())),
                    comptime!(1usize),
                ),
                self.nd_packed::<T, W>(comptime!(Guard::Checked)),
                comptime!(space.rank()),
            ),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                panic!("Tile::nd_split: this tile has no addressable N-D read surface")
            }
        }
    }
}

/// The tile's per-axis extents paired with its operand's mapping; the innermost is in lines.
#[cube]
pub(crate) fn axis_projection(
    #[comptime] space: Space,
    #[comptime] projection: Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
) -> ProjectionInKernel {
    let rank = comptime!(space.rank());
    let shape = Coords::constant(comptime!(line_extents(&space, vector_size, 0, rank)));

    ProjectionInKernel::new(shape, map, space, projection, vector_size)
}

/// The extents of `space` in `from..to`, the innermost rounded up to a line count.
pub(crate) fn line_extents(
    space: &Space,
    vector_size: usize,
    from: usize,
    to: usize,
) -> Vec<usize> {
    let last = space.rank() - 1;
    (from..to)
        .map(|p| {
            let e = space.extent_at(p);
            if p == last {
                e.div_ceil(vector_size)
            } else {
                e
            }
        })
        .collect()
}

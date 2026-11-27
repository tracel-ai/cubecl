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

/// Any [`LogicalLayout`] with an operand's [`Projection`] applied under it, each position read one
/// count of `walk` past where it is asked ([`walked`](Projected::walked)): where `walk` moves the
/// map only by constants, a read is the map at the asked position plus those constants, and two
/// reads that differ only in the walk share the map's evaluation.
///
/// # Assumptions
///
/// - A walked layout is read unmasked: its bounds are those of the asked position.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Projected<L: LogicalLayout> {
    inner: L,
    projection: ProjectionInKernel,
    #[cube(comptime)]
    walk: WalkUnits,
}

#[cube]
impl<L: LogicalLayout> Projected<L> {
    pub(crate) fn new(inner: L, projection: ProjectionInKernel) -> Self {
        Projected::<L> {
            inner,
            projection,
            walk: comptime!(WalkUnits::none()),
        }
    }

    /// This layout reading each position one count of `walk` past it.
    pub(crate) fn walked(self, #[comptime] walk: WalkUnits) -> Self {
        Projected::<L> {
            inner: self.inner,
            projection: self.projection,
            walk,
        }
    }

    /// The source position one count of the walk past `logical`.
    fn walked_from(&self, logical: CoordsDyn) -> CoordsDyn {
        let anchor = self.projection.anchor(logical, &self.walk);
        let once = Coords::constant(comptime!(vec![1; self.walk.dimensions.len()]));
        self.projection.advance(&anchor, &once, &self.walk)
    }
}

#[cube]
impl<L: LogicalLayout> Layout for Projected<L> {
    type Coordinates = L::Coordinates;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        self.walked_from(self.inner.to_source_pos(pos))
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (inner_pos, inner_in_bounds) = self.inner.to_source_pos_checked(pos);
        let proj_in_bounds = self.projection.is_in_bounds(inner_pos.clone());
        (
            self.walked_from(inner_pos),
            inner_in_bounds && proj_in_bounds,
        )
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
        let layout = g.dimension_projection(&space);
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
                let layout = g.dimension_projection(&self.place.space);
                g.packed::<T, W, CoordsDyn, ProjectionInKernel>(layout, guard)
            }
            TileKind::Procedural(data) => procedural_nd::<T, W>(data, &self.place.space, guard),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::nd: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::nd: a tma source has no element view"),
            TileKind::Lines(_) => {
                panic!("Tile::nd: the plane's units are read at a coordinate (`scale_at`)")
            }
        }
    }

    /// The raw words a packed tile holds, one coordinate per dimension of its
    /// [`Space`](crate::Space).
    pub(crate) fn nd_words<WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<u32, WP>, CoordsDyn> {
        let g = self.mem("nd_words");
        let layout = g.dimension_projection(&self.place.space);
        g.nd_words::<WP>(layout, guard)
    }

    /// [`nd_words`](Tile::nd_words) read as `I`, bound to `u32` at expansion.
    pub(crate) fn nd_words_as<I: Numeric, WP: Size>(
        &self,
        #[comptime] guard: Guard,
    ) -> Masked<'_, Vector<I, WP>, CoordsDyn> {
        let g = self.mem("nd_words_as");
        let layout = g.dimension_projection(&self.place.space);
        g.nd_words_as::<I, WP>(layout, guard)
    }

    /// The mutable twin of [`nd_words`](Tile::nd_words); refused where a write would alias.
    pub(crate) fn nd_words_mut<WP: Size>(&mut self) -> MaskedMut<'_, Vector<u32, WP>, CoordsDyn> {
        let space = comptime!(self.place.space.clone());
        let g = self.mem_mut("nd_words_mut");
        comptime!(assert!(
            g.projection.composition() != Composition::Overlapping,
            "Tile::nd_words_mut: an overlapping operand aliases under a write"
        ));
        let layout = g.dimension_projection(&space);
        g.nd_words_mut::<WP>(layout)
    }

    /// Whether [`Guard::Proved`] is sound here: false on a [`Boundary::Clamp`] dimension, and where
    /// a backward coefficient or a folded dimension's digits, which wrap inside the box, put its
    /// extremes off its two logical corners.
    pub(crate) fn guard_provable(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(data) => {
                comptime!(
                    !data.window.boundaries.contains(&Some(Boundary::Clamp))
                        && !data.projection.has_negative_scale()
                        && data.projection.digit_dimensions().is_empty()
                )
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(true),
        }
    }

    /// The map a read of this tile evaluates in the kernel: a memory window's
    /// ([`Memory::in_kernel_projection`]), any other tile's own [`projection`](Tile::projection).
    pub(crate) fn in_kernel_projection(&self) -> comptime_type!(Projection) {
        match &self.kind {
            TileKind::Memory(data) => data.in_kernel_projection(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => self.projection(),
        }
    }

    /// Whether a read through this tile's map lands on the pieces of a storage-tiled buffer
    /// ([`Memory::reads_pieces`]), rather than on its window's coordinates.
    pub(crate) fn reads_pieces(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(data) => data.reads_pieces(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether a read can be anchored once per run of `walk` and advanced from it, with the
    /// run's two ends bounding every position between them: an unpacked memory operand with no
    /// clamp, whose map the walk moves only by adds and holds
    /// ([`Projection::run_bounded_by_ends`]).
    pub(crate) fn run_anchorable(&self, #[comptime] walk: &WalkUnits) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(data) => {
                let projection = data.in_kernel_projection();
                comptime!(
                    matches!(data.store.packing, Packing::Plain)
                        && !data.window.boundaries.contains(&Some(Boundary::Clamp))
                        && projection.run_bounded_by_ends(walk)
                )
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }
}

/// A procedural tile's whole box as an N-D view, evaluated at the coordinate.
#[cube]
fn procedural_nd<'a, T: Numeric, W: Size>(
    data: &'a Procedural,
    #[comptime] space: &Space,
    #[comptime] guard: Guard,
) -> Masked<'a, Vector<T, W>, CoordsDyn> {
    let layout = dimension_projection(
        space,
        &comptime!(Projection::direct_over(space)),
        RuntimeMap::integral(space.rank()),
        comptime!(1usize),
    );
    Masked::new(
        View::<Vector<T, W>, CoordsDyn>::new::<&Procedural, CoordsDyn>(data, layout),
        comptime!(guard.checks() && data.bounds_check),
    )
}

/// An operand split into the map evaluated once per run and the physical view it addresses.
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
            TileKind::Memory(g) => {
                let map = g.dimension_projection(&space);
                let projection = g.in_kernel_projection();
                Gathered::new(
                    map,
                    // A caller that evaluated the map derived its coordinates itself, so both
                    // guards stay on.
                    g.packed::<T, W, CoordsDyn, CompactionStep>(
                        g.physical_box(),
                        comptime!(Guard::Checked),
                    ),
                    comptime!(projection.physical_rank()),
                )
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => Gathered::new(
                dimension_projection(
                    &space,
                    &comptime!(Projection::direct_over(&space)),
                    RuntimeMap::integral(space.rank()),
                    comptime!(1usize),
                ),
                self.nd_packed::<T, W>(comptime!(Guard::Checked)),
                space.rank(),
            ),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                panic!("Tile::nd_split: this tile has no addressable N-D read surface")
            }
        }
    }
}

/// The tile's per-dimension extents paired with its operand's mapping; the innermost is in lines.
#[cube]
pub(crate) fn dimension_projection(
    #[comptime] space: &Space,
    #[comptime] projection: &Projection,
    map: RuntimeMap,
    #[comptime] vector_size: usize,
) -> ProjectionInKernel {
    let rank = space.rank();
    let shape = Coords::constant(comptime!(line_extents(space, vector_size, 0, rank)));

    ProjectionInKernel::new(shape, map, space.clone(), projection.clone(), vector_size)
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

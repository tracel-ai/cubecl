//! The [`Tile`]: one operand's backing store plus the comptime [`Space`] it projects.

use cubecl::{prelude::*, unexpanded};

use cubecl_std::tensor::layout::CoordsDyn;

use cubecl::zspace::SmallVec;

use crate::stage::pipeline::payload::base::{StageOperand, StageSpec, stage_one};
use crate::*;

/// One operand's data: a runtime backing store and the comptime [`Space`] it projects.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Tile {
    pub(crate) kind: TileKind,
    /// The scales these values carry, innermost first: none until [`mul`](Tile::mul). A tile
    /// knows them whatever its kind and never applies them by itself.
    pub(crate) factor: Factor,
    /// Where this tile sits in its partitioning.
    #[cube(comptime)]
    pub(crate) place: Placement,
    /// The element it holds: its values', or the `u32` words a packed tile stores its fields in
    /// ([`read_element`](Tile::read_element) is what a reader takes them at).
    #[cube(comptime)]
    pub(crate) element: ElemType,
}

#[cube]
impl Tile {
    /// Pairs `kind` with its `place`, carrying no scales.
    pub(crate) fn new<T: Numeric>(kind: TileKind, #[comptime] place: Placement) -> Tile {
        Tile {
            kind,
            factor: Factor::none(),
            place,
            element: elem_type_of::<T>(),
        }
    }

    /// This tile's own box at this depth.
    pub fn space(&self) -> comptime_type!(Space) {
        comptime!(self.place.space.clone())
    }

    /// These cells placed at `depth` of `levels`: a tile whose grid the caller states.
    pub fn with_levels(&self, #[comptime] depth: usize, #[comptime] levels: Vec<Level>) -> Tile {
        Tile {
            kind: self.kind.clone(),
            factor: self.factor.clone(),
            place: comptime!(Placement::new(self.place.space.clone(), depth, levels)),
            element: comptime!(self.element),
        }
    }

    /// This tile's memory store; panics for kinds with none, naming `site`.
    pub(crate) fn mem(&self, #[comptime] site: &str) -> &Memory {
        match &self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source has no element view"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's units have no memory view")
            }
        }
    }

    /// The recipe this procedural tile evaluates; panics for every other kind.
    pub(crate) fn recipe(&self) -> &Procedural {
        match &self.kind {
            TileKind::Procedural(recipe) => recipe,
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => panic!("Tile::recipe: only a procedural tile has a recipe"),
        }
    }

    pub(crate) fn mem_mut(&mut self, #[comptime] site: &str) -> &mut Memory {
        match &mut self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// Who moves this operand's bytes into a stage. Panics on a plane fragment.
    pub fn delivery(&self) -> comptime_type!(Delivery) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.delivery),
            TileKind::TmaGmem(_) => comptime!(Delivery::Tma),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::delivery: a resident fragment is not a stage source")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => comptime!(Delivery::Procedural),
        }
    }

    /// Whether a cmma fragment draining into this tile goes through a scratch
    /// ([`FragmentDrain::Bounce`]): a destination that adds, overhangs, or has no address, or one
    /// whose element the fragment cannot be cast to in place (`in_place`, [`casts_in_place`]).
    pub(crate) fn fragments_bounce(&self, #[comptime] in_place: bool) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(m) => {
                let addressed = m.store.addressed();
                comptime!(
                    FragmentDrain::of(&m.access, addressed, in_place) == FragmentDrain::Bounce
                )
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    pub(crate) fn runtime_map(&self) -> RuntimeMap {
        match &self.kind {
            TileKind::Memory(g) => g.map.clone(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => RuntimeMap::integral(comptime!(self.place.space.rank())),
        }
    }

    /// The launch's cube size this operand was bound with, `0` when unknown.
    pub(crate) fn units(&self) -> comptime_type!(usize) {
        let fill = self.fill_units();
        comptime!(fill.count)
    }

    /// The units that share a cooperative fill of this tile ([`FillUnits`]): one plane's for a
    /// stage that plane owns, the cube's for everything else.
    pub(crate) fn fill_units(&self) -> comptime_type!(FillUnits) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.fill),
            TileKind::TmaGmem(t) => comptime!(FillUnits::cube(t.units)),
            TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => comptime!(FillUnits::cube(0)),
        }
    }

    /// Physical vector width of the backing store; `1` for a fragment or tma source. Refuses a
    /// memory whose loads span several axes: its reader asks for the vector tile instead.
    pub fn vector_size(&self) -> comptime_type!(usize) {
        match &self.kind {
            TileKind::Memory(d) => {
                let load = d.vector_tile(&comptime!(self.place.space.clone()));
                comptime!(load.along_one_axis("a reader asking Tile::vector_size").1)
            }
            TileKind::Lines(c) => c.line(),
            // One cell the descriptor moves a line: a word of a packed field's values.
            TileKind::TmaGmem(t) => comptime!(t.packing.served(1)),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::Procedural(_) => {
                comptime!(1usize)
            }
        }
    }

    /// What one vector load of this tile covers: a run along the innermost axis, or the stored
    /// tiles that hold [`vector_size`](Tile::vector_size) values ([`VectorTile::new`]).
    pub(crate) fn vector_tile(&self) -> comptime_type!(VectorTile) {
        let space = comptime!(self.place.space.clone());
        match &self.kind {
            TileKind::Memory(d) => d.vector_tile(&space),
            TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                let width = self.vector_size();
                comptime!(VectorTile::run(space.axis_at(space.rank() - 1), width))
            }
        }
    }

    /// Ask this accumulator to start from `init_from`; returns what took. A folding destination
    /// always answers [`Identity`](InitFrom::Identity).
    pub(crate) fn request_init_from(
        &mut self,
        #[comptime] init_from: InitFrom,
    ) -> comptime_type!(InitFrom) {
        match &mut self.kind {
            TileKind::Memory(d) => {
                let init_from = comptime!(match d.access.write {
                    Write::Accumulate | Write::Exclusive(_) => InitFrom::Identity,
                    Write::Replace => init_from,
                });
                d.set_init_from(comptime!(init_from));
                comptime!(init_from)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(InitFrom::Cell)
            }
        }
    }

    /// What a stage of this operand holds ([`StageElement`]): the values its fill decodes
    /// ([`decodes`](Tile::decodes)), or what its kind stages.
    pub(crate) fn stage_element(&self) -> comptime_type!(StageElement) {
        let decodes = self.decodes();
        let staged = match &self.kind {
            TileKind::Memory(d) => d.stage_element(),
            TileKind::TmaGmem(t) => comptime!(match t.packing {
                Packing::Plain => StageElement::Served,
                Packing::Packed { .. } => StageElement::Stored,
            }),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(StageElement::Served)
            }
        };
        comptime!(match decodes {
            true => StageElement::Decoded,
            false => staged,
        })
    }

    /// The element a reader takes these values at where it names none: what the tile holds, or
    /// `f32` for the fields a packed tile decodes ([`dequantized`](Tile::dequantized) names
    /// another).
    pub(crate) fn read_element(&self) -> comptime_type!(ElemType) {
        let packing = self.packing();
        comptime!(match packing {
            Packing::Plain => self.element,
            Packing::Packed { .. } => f32::elem_type_native(),
        })
    }

    /// The element a stage of this tile holds: the values its fill decodes, at `decoded` or
    /// [`read_element`](Tile::read_element), or what the tile holds.
    pub(crate) fn staged_element(
        &self,
        #[comptime] decoded: Option<ElemType>,
    ) -> comptime_type!(ElemType) {
        let staged = self.stage_element();
        let read = self.read_element();
        comptime!(match staged {
            StageElement::Decoded => decoded.unwrap_or(read),
            StageElement::Served | StageElement::Stored => self.element,
        })
    }

    /// How this tile's values sit in memory; see [`Memory::packing`].
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        match &self.kind {
            TileKind::Memory(d) => d.packing(),
            TileKind::Lines(c) => c.packing(),
            TileKind::TmaGmem(t) => comptime!(t.packing),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::Procedural(_) => {
                comptime!(Packing::Plain)
            }
        }
    }

    /// Whether this tile's mapping has overlapping windows, so only an N-D view describes it.
    pub fn gathered(&self) -> comptime_type!(bool) {
        let projection = self.projection();
        comptime!(projection.composition() == Composition::Overlapping)
    }

    /// Whether this tile evaluates values from coordinates instead of a backing buffer.
    pub(crate) fn is_procedural(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Procedural(_) => comptime!(true),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether this tile is a memory window.
    pub(crate) fn is_memory(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(_) => comptime!(true),
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether this tile is backed by shared memory, the only space a `sync_cube()` orders.
    pub fn is_shared(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(m) => comptime!(m.address == AddressSpace::Shared),
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// `Some(n)` for a recipe stating `n` separable factors, `None` otherwise.
    pub(crate) fn factors(&self) -> comptime_type!(Option<usize>) {
        match &self.kind {
            TileKind::Procedural(data) => data.factorization(),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(None),
        }
    }

    pub(crate) fn window_signed(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.signed),
            _ => comptime!(false),
        }
    }

    /// This operand's per-axis boundary handling, which a stage filled from it inherits.
    pub(crate) fn window_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.boundaries.clone()),
            _ => comptime!(SmallVec::new()),
        }
    }

    /// The boundaries factor-local normalization tests, kept on a stage's source window.
    pub(crate) fn separable_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
            TileKind::Memory(d) =>
            {
                #[comptime]
                match &d.source_window {
                    ComptimeOption::Some(source) => comptime!(source.boundaries.clone()),
                    ComptimeOption::None => comptime!(match d.address {
                        AddressSpace::Global => d.window.boundaries.clone(),
                        AddressSpace::Shared => SmallVec::new(),
                    }),
                }
            }
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(SmallVec::new()),
        }
    }

    /// The separable factor-normalization request of a procedural tile, if any.
    pub(crate) fn factor_normalization(&self) -> comptime_type!(Option<Normalization>) {
        match &self.kind {
            TileKind::Procedural(data) => comptime!(data.normalization.clone()),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(None),
        }
    }

    /// One factor of a separable recipe at `pos`; requires [`factors`](Tile::factors) to be `Some`.
    pub(crate) fn separable_factor<T: Numeric>(
        &self,
        pos: CoordsDyn,
        #[comptime] factor: usize,
    ) -> T {
        match &self.kind {
            TileKind::Procedural(data) => {
                data.read_factor_at::<T>(&pos, factor, comptime!(self.place.space.clone()))
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::separable_factor: a tile read from a buffer states no factorization, \
                     so `factors` answered `None` and there is no factor {factor} to evaluate"
                )
            }
        }
    }

    /// Whether this tile can state `axis`'s runtime size.
    pub(crate) fn witnesses(&self, #[comptime] axis: Axis) -> comptime_type!(bool) {
        let bounded = self.bounded();
        let projection = self.projection();
        comptime!(
            bounded
                && self.place.space.contains(axis)
                && self.place.space.is_dynamic(axis)
                && Witness::new(&projection, axis).is_some()
        )
    }

    /// How this tile's logical axes address its buffer's physical ones.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        match &self.kind {
            TileKind::Memory(g) => comptime!(g.projection.clone()),
            TileKind::Lines(c) => c.projection(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(Projection::direct_over(&self.place.space))
            }
        }
    }

    /// This tile at `like`'s place in its nest.
    pub(crate) fn nested_like(self, like: &Tile) -> Tile {
        Tile {
            kind: self.kind,
            factor: self.factor,
            place: comptime!(Placement::new(
                self.place.space.clone(),
                like.place.depth,
                like.place.levels.clone()
            )),
            element: comptime!(self.element),
        }
    }

    /// This operand's window at `from` on `axis`, reading zero past `until` (both in elements).
    pub fn within(&self, #[comptime] axis: Axis, from: usize, until: usize) -> Tile {
        let tile_kind = match &self.kind {
            TileKind::Memory(g) => TileKind::new_Memory(g.within(axis, from, until)),
            TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => panic!(
                "Tile::within: only a memory operand carries a window to place; a fragment, a \
                 tensor-map source and a recipe have none"
            ),
        };
        Tile {
            kind: tile_kind,
            factor: self.factor.clone(),
            place: comptime!(self.place.clone()),
            element: comptime!(self.element),
        }
    }

    /// Cuts `region` and returns the page table entry that window was cut with, loaded from the
    /// table unless an entry handed to this tile already named it. It is [`PageEntry::unlisted`]
    /// for a tile not read through a page table.
    pub fn read_page_entry(&self, region: &Region) -> PageEntry {
        let cut = self.at(region);
        match &cut.kind {
            TileKind::Memory(g) => g.layout.pinned_entry(),
            TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => PageEntry::unlisted(),
        }
    }

    /// This tile, its cuts taking their page from `entry` where they name its sequence and column,
    /// and from the table otherwise. `entry` must come from a tile reading the same table, keys
    /// handing theirs to values included: it is neither checked nor clamped to the pool, so an
    /// entry from another table reads the page that table lists. A tile not read through a page
    /// table is unchanged.
    pub fn with_page_entry(&self, entry: &PageEntry) -> Tile {
        let kind = match &self.kind {
            TileKind::Memory(g) => TileKind::new_Memory(g.with_page_entry(entry)),
            TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => self.kind.clone(),
        };
        Tile {
            kind,
            factor: self.factor.clone(),
            place: comptime!(self.place.clone()),
            element: comptime!(self.element),
        }
    }

    /// Window this tile down to `region`, no copy.
    pub fn at(&self, region: &Region) -> Tile {
        let skip = comptime!({
            let path = &region.path;
            assert!(
                path.base() <= self.place.depth && self.place.depth < path.depth(),
                "Tile::at: this tile sits {} levels down its nest, and the region's path runs \
                 from depth {} through {} levels; state the loops the tile is missing above it",
                self.place.depth,
                path.base(),
                path.len()
            );
            self.place.depth - path.base()
        });
        self.at_from(region, skip)
    }

    fn at_from(&self, region: &Region, #[comptime] i: usize) -> Tile {
        let sub = self.at_step(&region.step(i));
        if comptime!(i + 1 == region.path.len()) {
            sub
        } else {
            sub.at_from(region, comptime!(i + 1))
        }
    }

    /// One level down: window this tile to `step`'s box.
    pub(crate) fn at_step(&self, step: &Step) -> Tile {
        let tile_kind = match &self.kind {
            TileKind::Memory(g) => {
                TileKind::new_Memory(g.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::TmaGmem(t) => {
                TileKind::new_TmaGmem(t.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::Procedural(p) => {
                TileKind::new_Procedural(p.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::Lines(c) => {
                TileKind::new_Lines(c.at(step, comptime!(self.place.space.clone())))
            }
            // A plane tile passes through; legal only where the level cuts nothing on m/n.
            TileKind::PlaneTile(t) => {
                comptime!(assert!(
                    !MatrixGrid::new(&step.level, &self.place.space).cuts(),
                    "Tile::at: a level that cuts tiles cannot select into a single plane \
                     tile (it needs a partition, or a memory output)"
                ));
                TileKind::new_PlaneTile(t.clone())
            }
            TileKind::PlanePartition(p) => p.at_step(step, comptime!(self.place.space.clone())),
        };
        let windowed = Tile {
            kind: tile_kind,
            factor: self.factor.at(step),
            place: comptime!(self.place.at_step(&step.level)),
            element: comptime!(self.element),
        };
        windowed.indexed_at(step)
    }

    /// Whether this tile has a buffer bound to read an extent off.
    fn bounded(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(_) | TileKind::TmaGmem(_) => comptime!(true),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(false)
            }
        }
    }

    /// This operand's runtime logical size along `axis`; only for an axis it
    /// [`witnesses`](Tile::witnesses).
    pub(crate) fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let projection = self.projection();
        let p = comptime!(
            Witness::new(&projection, axis)
                .unwrap_or_else(|| panic!(
                    "Tile::runtime_extent: no bound of this operand is {axis:?}'s own extent (it \
                     gathers over it, or splits it across storage fragments); ask an operand that \
                     witnesses it"
                ))
                .dim()
        );
        let raw = match &self.kind {
            TileKind::Memory(g) => g.walked_bound(p).cast::<usize>(),
            TileKind::TmaGmem(t) => t.bound[p].cast::<usize>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::runtime_extent: a plane tile has no extent")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!(
                    "Tile::runtime_extent: a procedural tile and the plane's units have no extent"
                )
            }
        };
        // `bound` counts lines on the vectorized innermost axis.
        let last = comptime!(projection.physical_rank() - 1);
        let w = self.vector_size();
        comptime!(if p == last { w } else { 1usize }) * raw
    }

    /// This tile's own box as a runtime space, dynamic axes sized off its buffer.
    pub(crate) fn runtime_space(&self) -> Space {
        witnessed_space(comptime!(self.place.space.clone()), self, self, self)
    }

    /// The regions of the level below this tile over its own box.
    pub fn walk(&self) -> Walk {
        let space = self.runtime_space();
        Region::rooted(
            &space,
            comptime!(self.place.levels.clone()),
            comptime!(self.place.depth),
        )
        .walk()
    }

    /// The regions of `level` over this tile's own box.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        let space = self.runtime_space();
        Walk::of(
            &space,
            comptime!(level.clone()),
            Region::rooted(
                &space,
                comptime!(self.place.levels.clone()),
                comptime!(self.place.depth),
            ),
        )
    }

    /// Zero this whole tile.
    pub fn zero(&mut self) {
        let define!(Served) = self.element;
        match &mut self.kind {
            TileKind::Memory(d) => d.zero::<Served>(),
            TileKind::PlaneTile(t) => t.zero::<Served>(),
            TileKind::PlanePartition(p) => p.zero::<Served>(),
            TileKind::TmaGmem(_) => panic!("Tile::zero: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::zero: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// Seed this tile with `monoid`'s identity.
    pub(crate) fn init_identity<S: Numeric>(&mut self, #[comptime] monoid: Monoid) {
        match comptime!(monoid) {
            Monoid::Sum => self.zero(),
            Monoid::Prod | Monoid::Max | Monoid::Min => {
                self.init::<S>(Monoid::identity::<S>(monoid))
            }
        }
    }

    /// Multiply every partial this accumulator holds by `factor`, cast to its element.
    pub fn scale_by<E: Numeric>(&mut self, factor: E) {
        let define!(Served) = self.element;
        let factor = Served::cast_from(factor);
        match &mut self.kind {
            TileKind::PlaneTile(t) => t.scale::<Served>(factor),
            TileKind::PlanePartition(p) => p.scale::<Served>(factor),
            TileKind::Memory(_) => panic!(
                "Tile::scale_by: a memory tile is scaled by the cube, not by one unit (Tile::mul)"
            ),
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::scale_by: not writable")
            }
        }
    }

    /// The fragment grid this accumulator holds and one fragment's `m × n`.
    pub(crate) fn fragment_grid(&self) -> comptime_type!(((usize, usize), usize, usize)) {
        let edges = comptime!(MatrixAxes::edges(&self.place.space));
        let rows = comptime!(self.place.space.extent_at(edges.row_split));
        let cols = comptime!(self.place.space.extent_at(edges.col_split));
        match &self.kind {
            // A grid's own fragments, which a window shorter than them overhangs.
            TileKind::PlanePartition(p) => comptime!(((p.m_tiles, p.n_tiles), p.rows, p.cols)),
            TileKind::PlaneTile(_) => comptime!(((1, 1), rows, cols)),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::fragment_grid: only a plane-resident accumulator holds fragments")
            }
        }
    }

    /// Initialize this tile with `val`, cast to its element.
    pub fn init<E: Numeric>(&mut self, val: E) {
        let define!(Served) = self.element;
        let val = Served::cast_from(val);
        match &mut self.kind {
            TileKind::Memory(d) => d.init::<Served>(val),
            TileKind::PlaneTile(t) => t.init::<Served>(val),
            TileKind::PlanePartition(p) => p.init::<Served>(val),
            TileKind::TmaGmem(_) => panic!("Tile::init: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::init: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// A fresh tile shaped to stage one region of `level` of this operand, laid out as `storage`.
    pub fn stage(&self, #[comptime] level: Level, #[comptime] storage: StageStorage) -> Tile {
        let define!(Staged) = self.staged_element(comptime!(None));
        Memory::stage::<Staged>(
            &self.shaped_for(comptime!(level.clone())),
            level,
            storage,
            comptime!(None),
            comptime!(StageOwner::CUBE),
        )
    }

    /// A fresh stage of this operand for one region of `walk`, laid out as `storage`: shaped,
    /// owned and placed as the walk's stages are, the cube's or each plane's own copy. What a
    /// kernel stages once before a walk that leaves the operand unchanged, as an attention's
    /// query beside the keys it walks.
    ///
    /// A stage whose fill decodes its operand holds the values at `decoded`, or at `f32` where
    /// `None`, as [`Stages::smem`] does. An operand staged as it lies keeps its own element.
    pub fn stage_for(
        &self,
        walk: &Walk,
        #[comptime] storage: StageStorage,
        #[comptime] decoded: Option<ElemType>,
    ) -> Tile {
        let owner = walk.stage_owner();
        stage_one(
            self,
            comptime!(StageSpec {
                level: walk.level.clone(),
                depth: walk.depth(),
                storage,
                width: None,
                decoded,
                owner,
            }),
        )
    }

    /// A fresh shared-memory tile over `axes` of one region of `walk`, laid out as `storage` and
    /// served one value a line: placed where the walk's regions sit, and owned as the walk's
    /// stages are, one for the cube or a copy for each plane ([`Stages`]).
    ///
    /// What a kernel holds between two steps of its walk that no operand is: an attention's
    /// scores between its two contractions. Placed in the partitioning, it opens an accumulator
    /// and is windowed by the walk's regions like an operand.
    pub fn scratch<T: Numeric>(
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        #[comptime] storage: StageStorage,
    ) -> Tile {
        let owner = walk.stage_owner();
        let place = comptime!(Placement::new(
            walk.level.child(&walk.space).subspace(&axes),
            walk.depth(),
            walk.parent.path.root_levels(),
        ));
        let tile =
            Memory::smem_owned::<T>(place.space.clone(), 1usize, storage, 0usize, 0usize, owner);
        Tile::new::<T>(tile.kind, place)
    }

    /// A fresh shared-memory tile over `space`, laid out as `storage`, serving one value a line.
    pub fn shared<T: Numeric>(#[comptime] space: Space, #[comptime] storage: StageStorage) -> Tile {
        Memory::smem::<T>(space, comptime!(1usize), storage, comptime!(0usize))
    }

    /// A fresh shared-memory tile over `space`, served `vector_size` wide and filled by `units`
    /// units (`0`: the whole cube).
    pub fn smem<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
    ) -> Tile {
        Memory::smem::<T>(space, vector_size, storage, units)
    }

    /// A shared-memory tile over this tile's box, one element a line, its rows dense and in order,
    /// at this tile's place in its nest: what a bulk store moves out whole
    /// ([`copy_from`](Tile::copy_from) into a tile a tensor map backs), and what an accumulator
    /// drains into first, as it would into this tile. It starts on
    /// [`Delivery::TMA_STAGE_ALIGNMENT`], which a launch sizing its shared memory counts.
    pub fn smem_box(&self) -> Tile {
        let define!(Served) = self.element;
        let units = self.units();
        Memory::smem_owned::<Served>(
            comptime!(self.place.space.clone()),
            1usize,
            comptime!(StageStorage::Strided),
            units,
            comptime!(Delivery::TMA_STAGE_ALIGNMENT),
            comptime!(StageOwner::CUBE),
        )
        .nested_like(self)
    }

    /// Move `src` into `self`, the kind pairing picking the instruction.
    ///
    /// Into a tile a tensor map backs, from a shared-memory box ([`smem_box`](Tile::smem_box)),
    /// it is a bulk store that every unit of the cube reaches: each fences the writes it made to
    /// the box, the cube synchronizes, one unit stores it and waits for the engine to have read
    /// it, and the cube synchronizes again before anything writes the box.
    ///
    /// A source whose scales a stage keeps beside it copies its values as they are stored, into a
    /// stage that keeps them too; the stage's fill brings the scales. Any other scaled source is
    /// written with its scales applied, which only [`dequantize_from`](Tile::dequantize_from) does.
    pub fn copy_from(&mut self, src: &Tile) {
        let decodes = src.decodes();
        comptime!(assert!(
            !decodes,
            "Tile::copy_from: this source carries scales or a table that a copy as stored cannot \
             keep; write it with them applied (`Tile::dequantize_from`)"
        ));
        let (beside, kept) = (src.scales_beside(), self.scales_beside());
        comptime!(assert!(
            !beside || kept,
            "Tile::copy_from: this source's scales are kept beside its values, which only a stage \
             that keeps them too takes as they are stored"
        ));
        let define!(Served) = self.element;
        let define!(Source) = src.element;
        self.copy_unscaled_from::<Served, Source>(src);
    }

    /// Writes `src`'s numbers into this tile, every scale it carries applied and its table looked
    /// up: the one copy that applies a tile's scales.
    ///
    /// From memory, each load is decoded once, in the unit that reads it, its scales multiplied
    /// together in `f32` and the value rounded once. From an accumulator, it is the sum's drain,
    /// which multiplies the sum by the scale it carries; the sum is spent by its write.
    pub fn dequantize_from(&mut self, src: &Tile) {
        match &src.kind {
            TileKind::Memory(_) => {
                let into_memory = self.is_memory();
                comptime!(assert!(
                    into_memory,
                    "Tile::dequantize_from: values are decoded into memory; to load them into \
                     fragments, contract them: the fragment lands them"
                ));
                let define!(Served) = self.element;
                let define!(Source) = src.read_element();
                self.copy_scaled_from::<Served, Source>(src);
            }
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                let dest = self.clone();
                src.drained_into(&dest);
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => panic!(
                "Tile::dequantize_from: values are decoded from memory, and a sum from its \
                 fragments; a tensor-map source lands bytes as they lie, so stage it first"
            ),
        }
    }

    fn copy_unscaled_from<S: Numeric, R: Numeric>(&mut self, src: &Tile) {
        // Bound before the match, which borrows the kind.
        let space = comptime!(self.place.space.clone());
        match &src.kind {
            TileKind::PlanePartition(s) => match &mut self.kind {
                TileKind::Memory(d) => s.fragment().store_window::<S>(d, space),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("Tile::copy_from: a fragment stores into memory")
                }
            },
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => match (&mut self.kind, &src.kind) {
                (TileKind::Lines(d), TileKind::Memory(_)) => d.load::<R>(src),
                // A fragment loaded on its own holds `k` in the instruction's order.
                (TileKind::PlanePartition(d), TileKind::Memory(_)) => {
                    d.fill_from::<S>(src, DeepLoad::Steps)
                }
                (TileKind::PlaneTile(d), TileKind::Memory(_)) => {
                    d.load_window::<S>(src, DeepLoad::Steps)
                }
                (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.store_window::<S>(d, space),
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => s.load_into::<S>(d),
                (TileKind::TmaGmem(d), TileKind::Memory(s)) => d.store_from::<S>(s),
                (TileKind::Memory(d), TileKind::Memory(s)) => d.load_from::<S>(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => d.fill_procedural::<S>(s, space),
                (TileKind::PlaneTile(_), TileKind::PlaneTile(_)) => {
                    panic!("Tile::copy_from: plane tile to plane tile cast not wired")
                }
                _ => panic!("Tile::copy_from: unsupported kind pairing"),
            },
        }
    }

    /// Add a resident block `src` into this one, casting up to `T`, without touching memory.
    pub fn add_cast_from(&mut self, src: &Tile) {
        let define!(Served) = self.element;
        let define!(Source) = src.element;
        match (&mut self.kind, &src.kind) {
            (TileKind::PlaneTile(d), TileKind::PlaneTile(s)) => {
                d.add_cast_from::<Served, Source>(s)
            }
            (TileKind::PlanePartition(d), TileKind::PlanePartition(s)) => {
                // The fragment clones the block's array reference, so this promotes the block itself.
                let mut fragment = d.fragment();
                fragment.add_cast_from::<Served, Source>(&s.fragment())
            }
            _ => panic!(
                "Tile::add_cast_from: a resident block promotes into a resident block; memory \
                 is reached through `copy_cast_from`"
            ),
        }
    }

    /// [`copy_from`](Tile::copy_from) with a cast: stores a wider resident fragment down to `T`,
    /// or copies a memory window of the same box, the units sharing it a line at a time.
    pub fn copy_cast_from(&mut self, src: &Tile) {
        let define!(Served) = self.element;
        let define!(Source) = src.element;
        let space = comptime!(self.place.space.clone());
        let holder = comptime!(self.place.holder());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => {
                s.store_cast_window::<Source, Served>(d, space)
            }
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().store_cast_window::<Source, Served>(d, space)
            }
            // A window its holder shares is filled by the holder's units, whoever filled the
            // buffer it windows: a plane's window of the cube's scratch is the plane's.
            (TileKind::Memory(d), TileKind::Memory(s)) => d
                .filled_by(holder)
                .fill_cast_from::<Served, Source>(s, space),
            _ => panic!(
                "Tile::copy_cast_from: a fragment or a memory window stores into memory; nothing \
                 else casts"
            ),
        }
    }

    /// Spill this plane-resident tile into its slot of the plane's scratch.
    pub(crate) fn spill_to_scratch<S: Numeric>(&self) {
        match &self.kind {
            TileKind::PlaneTile(t) => t.spill_to_scratch::<S>(),
            TileKind::PlanePartition(p) => p.fragment().spill_to_scratch::<S>(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::spill_to_scratch: a plane-resident tile spills; nothing else does")
            }
        }
    }

    /// Add `src`'s spilled cells into this memory window.
    pub(crate) fn add_from_scratch<S: Numeric, R: Numeric>(&mut self, src: &Tile) {
        let space = comptime!(self.place.space.clone());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.add_from_scratch::<R, S>(d, space),
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().add_from_scratch::<R, S>(d, space)
            }
            _ => panic!("Tile::add_from_scratch: a spilled plane tile adds into memory"),
        }
    }

    /// Whether one factor tap lands inside the input axes that factor moves.
    pub(crate) fn separable_physical_tap_in_bounds(
        &self,
        pos: &CoordsDyn,
        #[comptime] axis: Axis,
    ) -> bool {
        match &self.kind {
            // A stage answers against the window it was filled from.
            TileKind::Memory(data) => {
                let projection = comptime!(data.projection.clone());
                if comptime!(!projection.addresses(axis)) {
                    true.runtime()
                } else {
                    #[comptime]
                    match &data.source_window {
                        ComptimeOption::Some(source) => source.axes_in_bounds(
                            &data.window.origin,
                            pos,
                            comptime!(projection.carriers(axis).to_vec()),
                        ),
                        ComptimeOption::None => match comptime!(data.address) {
                            AddressSpace::Global => data
                                .window
                                .axes_in_bounds(pos, comptime!(projection.carriers(axis).to_vec())),
                            AddressSpace::Shared => true.runtime(),
                        },
                    }
                }
            }
            TileKind::Procedural(data) => data.axis_in_bounds(pos, axis),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::separable_physical_tap_in_bounds: a separable gather needs an addressable rhs"
                )
            }
        }
    }
}

impl Tile {
    pub(crate) fn factor_dependencies(
        &self,
        _factors: Option<usize>,
        _row: Axis,
        _col: Axis,
    ) -> comptime_type!(Option<Vec<(bool, bool)>>) {
        unexpanded!()
    }
}

impl TileExpand {
    /// This tile's own box, readable inside `comptime!`.
    pub fn space(&self) -> Space {
        self.place.space.clone()
    }

    pub(crate) fn __expand_factor_dependencies_method(
        &self,
        scope: &Scope,
        factors: Option<usize>,
        row: Axis,
        col: Axis,
    ) -> Option<Vec<(bool, bool)>> {
        factors.map(|factors| match &self.kind {
            TileKindExpand::Procedural(data) => (0..factors)
                .map(|f| {
                    (
                        data.__expand_factor_reads_method(scope, f, row),
                        data.__expand_factor_reads_method(scope, f, col),
                    )
                })
                .collect(),
            TileKindExpand::Memory(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Lines(_) => (0..factors).map(|_| (true, true)).collect(),
        })
    }
}

/// `space` with each [`Dynamic`](crate::Extent) axis sized by the first of `a`, `b`, `c` that
/// [`witnesses`](Tile::witnesses) it.
#[cube]
pub(crate) fn witnessed_space(#[comptime] space: Space, a: &Tile, b: &Tile, c: &Tile) -> Space {
    let mut sizes = Sequence::<usize>::new();
    if comptime!(!space.is_static()) {
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            // Fold `Static` axes here so one `Dynamic` axis doesn't require a bound for the rest.
            let size = match comptime!(space.extent_raw(axis)) {
                Extent::Static(n) => comptime!(n).runtime(),
                Extent::Dynamic => {
                    let by_a = a.witnesses(axis);
                    let by_b = b.witnesses(axis);
                    let by_c = c.witnesses(axis);
                    if comptime!(by_a) {
                        a.runtime_extent(axis)
                    } else if comptime!(by_b) {
                        b.runtime_extent(axis)
                    } else if comptime!(by_c) {
                        c.runtime_extent(axis)
                    } else {
                        panic!(
                            "witnessed_space: {axis:?} is Dynamic and no operand states its size; \
                             every operand spanning it gathers over it, holds it Static, or is a \
                             fragment. Keep it Static in the kernel space, or give the operation \
                             an operand that maps it identically"
                        )
                    }
                }
            };
            sizes.push(size);
        }
    }
    Space::with_sizes(space, sizes)
}

/// Host-side stub of `for plane in tile`; never runs.
impl IntoIterator for Tile {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl IntoIterator for &Tile {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for plane in tile` iterates the level below the tile over its own box.
impl Iterable for TileExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let walk = self.__expand_walk_method(scope);
        if walk.const_len() == Some(1) {
            walk.expand_unroll(scope, body)
        } else {
            walk.expand(scope, body)
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.__expand_walk_method(scope).expand_unroll(scope, body)
    }
}

/// `for plane in &tile`.
impl Iterable for &TileExpand {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand_unroll(scope, body)
    }
}

impl Tile {
    /// These values under `scale`, multiplied in where they are read. Every axis `scale` varies
    /// along must be one of the values'; one it spans without varying along is read at its origin.
    pub fn mul(&self, _scale: &Tile) -> Tile {
        unexpanded!()
    }

    /// The factor a leaf reads these values under.
    pub(crate) fn reader(
        &self,
        _axes: MatrixAxes,
        _matrix: usize,
        _side: Side,
        _out: Space,
        _acc_axes: MatrixAxes,
    ) -> FactorReader {
        unexpanded!()
    }

    /// This tile without its outer scales that vary only along axes of `space`, and those scales,
    /// for a sum `instruction` contracts. With `space` the output's, they are the scales the sum
    /// takes out of every term and applies once ([`scaled`](Tile::scaled)), and the tile left
    /// carries the ones each term needs.
    ///
    /// Words of block scales always give up the scales above them: the instruction reads the
    /// words undecoded, and holds no other scale. A decoded tile gives them up where the sum is
    /// scaled in place ([`Instruction::scales_its_sum_in_place`]), since a load converting codes
    /// in lane runs takes one level of scales; under any other sum they ride the load, which
    /// multiplies by them as it decodes, rather than send the sum through shared memory.
    /// A tile indexing a table gives up none: only its copy decodes it.
    pub fn factor(&self, _space: Space, _instruction: Instruction) -> (Tile, Factor) {
        unexpanded!()
    }

    /// This tile under `factor`'s scales, outside its own.
    pub fn scaled(&self, _factor: &Factor) -> Tile {
        unexpanded!()
    }

    /// These values read placed: an `e2m1` field the device emulates decoded without its lift,
    /// which the factor this tile is read under carries instead ([`reader`](Tile::reader)), one
    /// multiply a reader rather than one a value. What a leaf that applies the factor to every
    /// value it reads takes; any other tile as it is.
    pub(crate) fn placed(&self) -> Tile {
        unexpanded!()
    }

    /// These values as indices into `table`, decoded only by [`copy_from`](Tile::copy_from) and
    /// [`decoded`](Tile::decoded): what an operand bound under a codebook
    /// ([`TileArgLaunch::codebook`]) is served as.
    #[allow(dead_code)] // Reached through its expand, from `TileArg::tile`.
    pub(crate) fn lookup(&self, _table: &Tile) -> Tile {
        unexpanded!()
    }

    pub(crate) fn refuse_factor(&self, _site: &str) {
        unexpanded!()
    }

    /// Whether these values carry scales ([`mul`](Tile::mul)) or index a table
    /// ([`lookup`](Tile::lookup)).
    pub(crate) fn quantized(&self) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// How many scales these values carry ([`mul`](Tile::mul)), down the chain.
    pub(crate) fn factor_levels(&self) -> comptime_type!(usize) {
        unexpanded!()
    }

    /// This tile without the scales it carries.
    pub(crate) fn unscaled(&self) -> Tile {
        unexpanded!()
    }

    /// The innermost level of the scales these values carry, alone: what a block-scaled
    /// instruction reads a scale of per block.
    pub(crate) fn innermost_factor(&self) -> Factor {
        unexpanded!()
    }

    /// Whether filling a stage from these values decodes them: they carry scales or a table, and
    /// not scales a stage keeps beside them.
    ///
    /// A stage keeps beside the values the words of block scales they carry: only the
    /// block-scaled instruction reads those, so the values are staged as they are stored and the
    /// scales staged with them.
    pub(crate) fn decodes(&self) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// Whether these values carry scales a stage keeps beside them ([`decodes`](Tile::decodes)).
    pub(crate) fn scales_beside(&self) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// This stage of `operand` with `operand`'s scales staged beside it as `spec` shapes it,
    /// where a stage keeps them beside it ([`decodes`](Tile::decodes)); itself otherwise.
    pub(crate) fn with_scales_staged(&self, _operand: &Tile, _spec: StageSpec) -> Tile {
        unexpanded!()
    }

    /// Fill the scales this stage keeps beside its values from `src`'s, under `meeting`; nothing
    /// where it keeps none.
    pub(crate) fn fill_scales_from(&self, _src: &Tile, _meeting: &Meeting) {
        unexpanded!()
    }

    /// Free the shared memory of the scales this stage keeps beside its values.
    pub(crate) fn free_scales(&self) {
        unexpanded!()
    }

    /// The scales a stage of these values keeps beside them, as a slot plans them; none where
    /// it keeps none.
    pub(crate) fn scale_operands(&self) -> comptime_type!(Vec<StageOperand>) {
        unexpanded!()
    }
}

impl TileExpand {
    pub fn __expand_mul_method(&self, scope: &Scope, scale: &TileExpand) -> TileExpand {
        let values = self.place.space.clone();
        let projection = scale.clone().__expand_projection_method(scope);
        // An axis the scales span without varying along it is read at its origin
        // ([`scale_for`](Tile::scale_for)): a scale broadcasts by omission.
        let varied: Vec<Axis> = scale
            .place
            .space
            .axes()
            .filter(|&axis| projection.addresses(axis))
            .collect();
        assert!(
            varied.iter().all(|&axis| values.contains(axis)),
            "Tile::mul: the scales vary along {varied:?} where the values span {:?}; a scale is \
             looked up at the value's coordinates, so every axis the scales vary along is one of \
             the values'",
            values.axes().collect::<Vec<_>>()
        );
        let mut out = self.clone();
        out.factor = out.factor.and(FactorExpand::of(scale));
        out
    }

    pub(crate) fn __expand_reader_method(
        &self,
        scope: &Scope,
        axes: MatrixAxes,
        matrix: NativeExpand<usize>,
        side: Side,
        out: Space,
        acc_axes: MatrixAxes,
    ) -> FactorReaderExpand {
        self.refuse_codebook("a leaf's read");
        let values = self.place.space.clone();
        // A line is the run of a load along the innermost axis; a load stored across several
        // columns is read as their runs, each placed at its own column.
        let vector_size = self.clone().__expand_vector_tile_method(scope).extents()[0].1;
        let factor = self.factor.clone();
        if let Some(inner) = factor.innermost_tile() {
            let projection = inner.clone().__expand_projection_method(scope);
            check_scales_omit_rather_than_divide(&projection);
            check_scales_ride(side, &inner.place.space, &out, acc_axes);
            let innermost = values.axis_at(values.rank() - 1);
            assert!(
                vector_size == 1 || !projection.addresses(innermost),
                "Tile::mul: a line runs {vector_size} values along {innermost:?}, which its \
                 scales address, so one line lies under several scales; serve the factor one \
                 value a line, or omit {innermost:?} from the scales"
            );
        }
        // A tile read placed carries its values' lift in its factor: one multiply of the coarse
        // levels, met once, and every value read under the factor even where it has no scale.
        let lift = self.__expand_packing_method(scope).lift();
        let coarse = match lift == 1.0 {
            true => factor.coarse(scope),
            false => {
                let lift: NativeExpand<f32> =
                    cubecl::ir::ExpandValue::constant((lift as u64).into(), f32::elem_type(scope))
                        .into();
                MulExpand::__expand_mul_method(factor.coarse(scope), scope, lift)
            }
        };
        FactorReaderExpand {
            coarse,
            inner: factor.innermost(),
            scaled: factor.scaled() || lift != 1.0,
            values,
            axes,
            vector_size,
            matrix,
        }
    }

    pub fn __expand_factor_method(
        &self,
        scope: &Scope,
        space: Space,
        instruction: Instruction,
    ) -> (TileExpand, FactorExpand) {
        let gives_up = !self.looks_up()
            && (self.factor.staged_beside(scope) || instruction.scales_its_sum_in_place());
        if !gives_up {
            return (self.clone(), FactorExpand::default());
        }
        let (inner, outer) = self.factor.split_on(scope, &space);
        let mut values = self.clone();
        values.factor = inner;
        (values, outer)
    }

    pub fn __expand_scaled_method(&self, _scope: &Scope, factor: &FactorExpand) -> TileExpand {
        let mut out = self.clone();
        out.factor = out.factor.and(factor.clone());
        out
    }

    pub(crate) fn __expand_placed_method(&self, _scope: &Scope) -> TileExpand {
        let mut out = self.clone();
        if let TileKindExpand::Memory(memory) = &mut out.kind {
            memory.store.packing = memory.store.packing.placed();
        }
        out
    }

    /// Whether these values index a table ([`Tile::lookup`]).
    fn looks_up(&self) -> bool {
        match &self.kind {
            TileKindExpand::Memory(memory) => memory.codebook.present(),
            _ => false,
        }
    }

    pub(crate) fn __expand_quantized_method(&self, _scope: &Scope) -> bool {
        self.looks_up() || self.factor.scaled()
    }

    pub(crate) fn __expand_factor_levels_method(&self, _scope: &Scope) -> usize {
        self.factor.depth()
    }

    pub(crate) fn __expand_unscaled_method(&self, _scope: &Scope) -> TileExpand {
        let mut out = self.clone();
        out.factor = FactorExpand::default();
        out
    }

    pub(crate) fn __expand_innermost_factor_method(&self, _scope: &Scope) -> FactorExpand {
        self.factor.innermost()
    }

    pub(crate) fn __expand_decodes_method(&self, scope: &Scope) -> bool {
        self.__expand_quantized_method(scope) && !self.__expand_scales_beside_method(scope)
    }

    pub(crate) fn __expand_scales_beside_method(&self, scope: &Scope) -> bool {
        !self.looks_up() && self.factor.staged_beside(scope)
    }

    pub(crate) fn __expand_with_scales_staged_method(
        &self,
        scope: &Scope,
        operand: &TileExpand,
        spec: StageSpec,
    ) -> TileExpand {
        let mut stage = self.clone();
        if operand.__expand_scales_beside_method(scope) {
            stage.factor = operand.factor.staged(scope, &spec);
        }
        stage
    }

    pub(crate) fn __expand_fill_scales_from_method(
        &self,
        scope: &Scope,
        src: &TileExpand,
        meeting: &MeetingExpand,
    ) {
        if self.factor.staged_beside(scope) {
            self.factor.fill_from(scope, &src.factor, meeting);
        }
    }

    pub(crate) fn __expand_scale_operands_method(&self, scope: &Scope) -> Vec<StageOperand> {
        // A table the values have yet to read places the scales beside them too.
        let keys = self.pending_keys();
        match self.__expand_scales_beside_method(scope) {
            true => self
                .factor
                .stage_operands(scope)
                .into_iter()
                .map(|scale| StageOperand {
                    keys: keys.clone(),
                    ..scale
                })
                .collect(),
            false => Vec::new(),
        }
    }

    pub(crate) fn __expand_free_scales_method(&self, scope: &Scope) {
        if self.factor.staged_beside(scope) {
            self.factor.free_stage(scope);
        }
    }

    pub fn __expand_lookup_method(&self, _scope: &Scope, table: &TileExpand) -> TileExpand {
        assert!(
            table.place.space.rank() == 1,
            "Tile::lookup: a table is a tile of one axis, its positions; this one spans {:?}",
            table.place.space.axes().collect::<Vec<_>>()
        );
        let mut out = self.clone();
        match &mut out.kind {
            TileKindExpand::Memory(memory) => {
                let packing = memory.store.packing;
                let Packing::Packed {
                    field: Field::Index { bits },
                } = packing
                else {
                    panic!(
                        "Tile::lookup: the values index a table through an index field \
                         (`Field::Index`), and these are stored as {packing:?}"
                    )
                };
                assert!(
                    table.place.space.extent_at(0) >= 1 << bits,
                    "Tile::lookup: a {bits}-bit index names {} positions; the table holds {}",
                    1usize << bits,
                    table.place.space.extent_at(0)
                );
                assert!(
                    !memory.codebook.present(),
                    "Tile::lookup: these values already index a table"
                );
                memory.codebook = CodebookExpand::of(table);
            }
            TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Procedural(_)
            | TileKindExpand::Lines(_) => {
                panic!("Tile::lookup: a table is indexed by values stored in memory")
            }
        }
        out
    }

    pub(crate) fn __expand_refuse_factor_method(&self, _scope: &Scope, site: &str) {
        self.refuse_codebook(site);
        if let TileKindExpand::Memory(_) = &self.kind {
            assert!(
                !self.factor.scaled(),
                "{site}: this leaf takes its operands from registers, where scales have nowhere                  to land; contract through a fragment or in memory"
            );
        }
    }

    /// Refuses values that index a table ([`Tile::lookup`]) at `site`.
    fn refuse_codebook(&self, site: &str) {
        if let TileKindExpand::Memory(memory) = &self.kind {
            assert!(
                !memory.codebook.present(),
                "{site}: these values are indices into a codebook (`TileArgLaunch::codebook`), \
                 which only a copy or a decoded read decodes; copy them into a stage first \
                 (`stage.copy_from(&w)`), or read them `decoded`"
            );
        }
    }
}

#[cube]
impl Tile {
    /// The one scale at `coords`, one entry per axis of this tile's space, read from memory under
    /// `guard`.
    #[allow(dead_code)] // Reached through its expand, from [`Factor`].
    pub(crate) fn scale_at<S: Numeric>(&self, coords: &Coords<u32>, #[comptime] guard: Guard) -> S {
        match &self.kind {
            TileKind::Lines(lines) => lines.read::<S>(coords),
            TileKind::Memory(_) => self.value_in_line::<S>(coords, guard),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::scale_at: a scale is read from memory or from the plane's units")
            }
        }
    }

    /// The one scale covering the value at `coords` of a tile spanning `values`. An axis of the
    /// scales the values do not span is one the scales do not vary along: a sum's scale names the
    /// operand's axes, and is read at the origin of the ones the sum has summed away.
    #[allow(dead_code)] // Reached through its expand, from [`Factor`].
    pub(crate) fn scale_for<S: Numeric>(
        &self,
        coords: &Coords<u32>,
        #[comptime] values: Space,
        #[comptime] guard: Guard,
    ) -> S {
        let rank = comptime!(self.place.space.rank());
        let projection = self.projection();
        let mut own = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(self.place.space.axis_at(p));
            if comptime!(values.contains(axis)) {
                own.push(coords.at(comptime!(values.position(axis))));
            } else {
                comptime!(assert!(
                    !projection.addresses(axis),
                    "Tile::scale_for: these scales vary along {axis:?}, which the values they \
                     scale do not span"
                ));
                own.push(0u32);
            }
        }
        self.scale_at::<S>(&own, guard)
    }

    /// The one value at `coords` of a tile that serves lines: the load holding it, read whole,
    /// and the value's place in it, whatever axes the load spans. A packed load decodes only the
    /// field that value is, not the whole line.
    #[allow(dead_code)] // Reached through its expand, from [`Tile::scale_at`].
    fn value_in_line<S: Numeric>(&self, coords: &Coords<u32>, #[comptime] guard: Guard) -> S {
        let define!(Served) = self.element;
        let space = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let load = self.vector_tile();
        let width = comptime!(load.values());
        let packing = self.packing();
        let mut position = 0u32.runtime();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let extent = comptime!(load.extent_along(axis) as u32);
            if comptime!(extent > 1) {
                let step = comptime!(load.step_along(axis) as u32);
                position += coords.at(p).remainder(extent).times(step);
            }
        }
        match comptime!(packing) {
            Packing::Plain => {
                let size!(W) = width;
                let line = self
                    .nd_packed::<Served, W>(guard)
                    .read(load.index(coords, &space));
                let value = if comptime!(width > 1) {
                    line.extract_dynamic(position.cast::<usize>())
                } else {
                    line.extract(0usize)
                };
                S::cast_from(value)
            }
            Packing::Packed { field } => {
                let (bits, per_word) =
                    comptime!((field.size_bits() as u32, field.per_word() as u32));
                let words_wide = comptime!(packing.physical(width));
                let size!(WP) = words_wide;
                let words = self.nd_words::<WP>(guard).read(load.index(coords, &space));
                let word = if comptime!(words_wide > 1) {
                    words.extract_dynamic((position / per_word).cast::<usize>())
                } else {
                    words.extract(0usize)
                };
                let shifted = word >> ((position % per_word) * bits);
                unpack_line::<S, Const<1>, Const<1>>(Vector::new(shifted), field).extract(0usize)
            }
        }
    }
}

#[cube]
impl Tile {
    /// This tile's cells in order, one value a slice of something else, as the rows of a softmax:
    /// a window of memory a plane holds, `count` cells, every unit of the plane reading them all.
    pub(crate) fn cells<E: Float>(&self, #[comptime] count: usize) -> Array<E> {
        comptime!(assert!(
            self.place.holder() == ComputeScope::Plane && self.place.space.cells() == count,
            "Tile::cells: a plane's window of {count} cells, not {:?} held by {:?}",
            self.place.space,
            self.place.holder()
        ));
        match &self.kind {
            TileKind::Memory(window) => window.cells(count),
            TileKind::PlanePartition(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => Tile::refuse_cells(),
        }
    }

    /// Write `values` over this tile's `count` cells ([`cells`](Tile::cells)), from the plane's
    /// first unit.
    pub(crate) fn set_cells<E: Float>(&mut self, values: &Array<E>, #[comptime] count: usize) {
        comptime!(assert!(
            self.place.holder() == ComputeScope::Plane && self.place.space.cells() == count,
            "Tile::cells: a plane's window of {count} cells, not {:?} held by {:?}",
            self.place.space,
            self.place.holder()
        ));
        match &mut self.kind {
            TileKind::Memory(window) => window.set_cells(values, count),
            TileKind::PlanePartition(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => Tile::refuse_cells(),
        }
    }

    fn refuse_cells() -> ! {
        panic!("Tile::cells: a plane's window of memory holds the cells every unit of it reads")
    }
}

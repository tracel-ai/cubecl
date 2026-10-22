//! Allocating the tile an operand is staged into, and filling a stage from a procedural source.

use cubecl::prelude::*;

use crate::*;

/// What a stage of an operand holds, and so how its fill lands it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StageElement {
    /// The values the operand serves, which are what it stores: copied as they lie.
    Served,
    /// The values the operand serves, decoded by the fill: from scales, a codebook, or packed
    /// fields as wide as their word, which take the same shared memory staged either way.
    Decoded,
    /// The stored form, words of several packed fields: copied as they lie.
    Stored,
}

impl StageElement {
    /// Whether the stage holds the values the operand serves, rather than its words.
    pub(crate) fn holds_values(self) -> bool {
        self != Self::Stored
    }
}

#[cube]
impl Memory {
    /// Cooperatively materialize a procedural source into this plain, direct scalar tile.
    /// Every unit that shares its fill ([`FillUnits`]) must own this window.
    pub(crate) fn fill_procedural<T: Numeric>(
        &mut self,
        src: &Procedural,
        #[comptime] space: Space,
    ) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "Memory::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Runtime window, so a Dynamic extent witnessed by another operand still works.
        let shape = self.window.extent.clone();
        let fill = comptime!(self.access.fill);
        let mut dst = self.flat_mut::<T, Const<1>>();
        let total = dst.shape();
        let stride = FillUnits::workers(fill);
        let mut i = FillUnits::worker(fill);
        while i < total {
            let pos = shape.unravel(i.cast::<u32>());
            // TODO: masked cells use Procedural's zero fallback, not the reduction's identity.
            dst.write(
                i,
                Vector::cast_from(src.read_masked::<T>(&pos, comptime!(space.clone()))),
            );
            i += stride;
        }
    }

    /// Allocate a tile staging one `divide()` sub-tile of `operand`, laid out as `storage` and
    /// served `width` wide where stated, for `owner`: the cube, or each plane a copy of its own,
    /// which only a plain stage of served values is.
    pub(crate) fn stage<S: Numeric>(
        operand: &Tile,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        match comptime!(storage.clone()) {
            // The plane holds the units at the depth of one region of `level`.
            StageStorage::Lines { read } => {
                comptime!(refuse_for_a_plane(owner, "held in the plane's units"));
                Tile {
                    kind: TileKind::new_Lines(Lines::new(
                        operand,
                        comptime!(level.clone()),
                        comptime!(read),
                    )),
                    factor: Factor::none(),
                    place: comptime!(Placement::new(
                        level.child(&operand.place.space),
                        operand.place.depth + 1,
                        operand.place.levels.clone()
                    )),
                    element: elem_type_of::<S>(),
                }
            }
            _ => Memory::stage_memory::<S>(operand, level, storage, width, owner),
        }
    }

    /// [`stage`](Memory::stage) in shared memory.
    fn stage_memory<S: Numeric>(
        operand: &Tile,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let element = operand.stage_element();
        let decodes = comptime!(element == StageElement::Decoded);
        match comptime!(element) {
            StageElement::Served | StageElement::Decoded => {
                let space = comptime!(level.child(&operand.place.space));
                let projection = operand.projection();
                let units = operand.units();
                let source_width = operand.vector_size();
                let vector_size = comptime!(match width {
                    Some(width) => {
                        assert!(
                            source_width == 1,
                            "Memory::stage: a padded stage assembles its lines from scalar \
                             source cells, so the operand it pads must be unvectorized (it is \
                             served {source_width} wide)"
                        );
                        assert!(
                            width > 1,
                            "Memory::stage: a padded stage width must widen the operand's own \
                             1-wide lines (got {width})"
                        );
                        width
                    }
                    // A decoding fill writes the stage value by value, so its lines need only
                    // divide the source's loads: as wide as every region below the stage cuts
                    // whole along its innermost axis, or a step would start mid-line.
                    None if decodes => {
                        let innermost = space.axis_at(space.rank() - 1);
                        space
                            .leaf(operand.place.below_level(&level))
                            .whole_run(innermost, source_width)
                    }
                    None => source_width,
                });
                // A TMA-filled stage's buffer must be TMA-aligned; TMA operands are always direct.
                let delivery = operand.delivery();
                let alignment = comptime!(match delivery.is_tma() {
                    true => Delivery::TMA_STAGE_ALIGNMENT,
                    false => 0usize,
                });
                // A decoded stage is dense over the values' axes whatever the source's map.
                if comptime!(projection.is_direct() || decodes) {
                    Memory::smem_owned::<S>(space, vector_size, storage, units, alignment, owner)
                } else {
                    comptime!(refuse_for_a_plane(owner, "gathered"));
                    Memory::smem_gathered::<S>(
                        space,
                        vector_size,
                        storage,
                        units,
                        projection,
                        &operand.runtime_map(),
                        operand.window_signed(),
                        operand.window_boundaries(),
                        owner,
                    )
                }
            }
            StageElement::Stored => {
                comptime!(refuse_for_a_plane(owner, "kept in its stored form"));
                Memory::smem_stored::<S>(operand, level, storage, owner)
            }
        }
    }

    /// [`stage`](Memory::stage) in the element the operand is stored in, for the cube `owner`
    /// names, whose units fill it.
    fn smem_stored<S: Numeric>(
        operand: &Tile,
        #[comptime] level: Level,
        #[comptime] storage: StageStorage,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let space = comptime!(level.child(&operand.place.space));
        let vector_size = operand.vector_size();
        let units = operand.units();
        // A box lands where the descriptor's swizzle span starts.
        let (packing, alignment) = match &operand.kind {
            TileKind::Memory(g) => comptime!((g.store.packing, 0usize)),
            TileKind::TmaGmem(t) => comptime!((t.packing, Delivery::TMA_STAGE_ALIGNMENT)),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Memory::smem_stored: a fragment is not a stage source")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!(
                    "Memory::smem_stored: a procedural tile and the plane's units are not a stage source"
                )
            }
        };
        match comptime!(packing) {
            Packing::Plain => {
                // The values the bulk engine lands come packed; a plain stored stage keeps the
                // copy's alignment.
                Memory::smem_owned::<S>(space, vector_size, storage, units, 0usize, owner)
            }
            packing => Memory::smem_packed::<S>(
                space,
                vector_size,
                storage,
                units,
                packing,
                alignment,
                owner,
            ),
        }
    }
}

/// Refuse a plane its own copy of a stage `form` describes: only a plain shared-memory stage of
/// served values is laid out one copy per plane.
fn refuse_for_a_plane(owner: StageOwner, form: &str) {
    assert!(
        matches!(owner, StageOwner::Cube { .. }),
        "Memory::stage: this walk hands each plane a stage of its own, which is laid out only for \
         a plain shared-memory stage of served values, and this one is {form}; stage the operand \
         plain, or walk it from the cube"
    );
}

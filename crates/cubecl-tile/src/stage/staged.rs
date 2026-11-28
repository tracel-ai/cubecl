//! Allocating the tile an operand is staged into, and filling a stage from a procedural source.

use cubecl::prelude::*;

use crate::*;

/// What a stage of an operand holds, and so how its fill lands it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StageElement {
    /// The values the operand serves, which are what it stores: copied as they lie.
    Served,
    /// The values the operand serves, decoded by the fill: from scales, a codebook, packed fields
    /// as wide as their word, which take the same shared memory staged either way, or narrower
    /// ones for a reader that names the element it decodes them into.
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
    /// Every unit its fill's level names ([`Level::copied_by`]) must own this window.
    pub(crate) fn fill_procedural<T: Numeric>(
        &mut self,
        src: &Procedural,
        #[comptime] space: &Space,
    ) {
        comptime!(assert!(
            self.store.packing.as_stored()
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "Memory::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Runtime window, so a Dynamic extent witnessed by another operand still works.
        let shape = self.window.extent.clone();
        let fill = comptime!(self.access.copy.clone());
        let mut dst = self.flat_mut::<T, Const<1>>();
        let total = dst.shape();
        for line in Walk::copy(&fill, total) {
            let i = line.copied_line();
            let pos = shape.unravel(i.cast::<u32>());
            // TODO: masked cells use Procedural's zero fallback, not the reduction's identity.
            dst.write(i, Vector::cast_from(src.read_masked::<T>(&pos, space)));
        }
    }

    /// Allocate a tile staging one `divide()` sub-tile of `operand`, laid out as `storage` and
    /// served `width` wide where stated, for `owner`: the cube, or each plane a copy of its own,
    /// which only a plain stage of served values is. A reader that names the element it decodes
    /// into, `decoded`, is staged values where it could be staged words
    /// ([`stage_element_for`](Tile::stage_element_for)). A plain operand staged as it lies is laid
    /// out in the order the fragments reading it take, `contracted` the Dimension its caller
    /// states a contraction sums it along ([`StageStorage::ordered_for_fragments`]).
    pub(crate) fn stage<S: Numeric>(
        operand: &Tile,
        #[comptime] level: &Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] owner: StageOwner,
        #[comptime] decoded: Option<ElemType>,
        #[comptime] contracted: Option<Dimension>,
    ) -> Tile {
        match storage.clone() {
            // The plane holds the units at the depth of one region of `level`.
            StageStorage::Lines { read } => {
                owner.refuse_for_a_plane("held in the plane's units");
                Tile {
                    kind: TileKind::new_Lines(Lines::new(operand, level, comptime!(read))),
                    factor: Factor::none(),
                    place: comptime!(Placement::new(
                        level.child(&operand.place.space),
                        operand.place.depth + 1,
                        operand.place.levels.clone()
                    )),
                    element: elem_type_of::<S>(),
                }
            }
            _ => Memory::stage_memory::<S>(
                operand, level, storage, width, owner, decoded, contracted,
            ),
        }
    }

    /// [`stage`](Memory::stage) in shared memory.
    fn stage_memory<S: Numeric>(
        operand: &Tile,
        #[comptime] level: &Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] owner: StageOwner,
        #[comptime] decoded: Option<ElemType>,
        #[comptime] contracted: Option<Dimension>,
    ) -> Tile {
        let element = operand.stage_element_for(decoded);
        let decodes = comptime!(element == StageElement::Decoded);
        let packing = operand.packing();
        let space = comptime!(level.child(&operand.place.space));
        let factor = Memory::factor_stage::<S>(operand, &space, width, element, contracted);
        let target = comptime::target_properties().comptime();
        let storage = comptime!(storage.ordered_for_fragments(&factor, &target.mma));
        let stage = Memory::stage_memory_as_served::<S>(
            operand, level, storage, width, owner, element, decodes,
        );
        // Words are served as stored, so their stage holds them as they lie, still words.
        match comptime!(packing) {
            Packing::Words { .. } => stage.served_as(packing),
            Packing::Plain | Packing::Packed { .. } => stage,
        }
    }

    /// The facts of `operand`'s stage over `space` that the order its fragments read it in is
    /// decided by, holding `element` and served `width` wide where stated, `contracted` the
    /// Dimension its caller states a contraction sums it along.
    fn factor_stage<S: Numeric>(
        operand: &Tile,
        #[comptime] space: &Space,
        #[comptime] width: Option<usize>,
        #[comptime] element: StageElement,
        #[comptime] contracted: Option<Dimension>,
    ) -> comptime_type!(FactorStage) {
        let packing = operand.packing();
        let quantized = operand.quantized();
        let memory = operand.is_memory();
        let projection = operand.projection();
        let delivery = operand.delivery();
        let source_width = operand.vector_size();
        let elem = elem_type_of::<S>();
        comptime!(FactorStage {
            space: space.clone(),
            contracted,
            elem_bits: elem.size_bits(),
            vector_size: source_width,
            delivery,
            source: StageSource {
                served: element == StageElement::Served,
                plain: packing == Packing::Plain && !quantized,
                own_width: width.is_none(),
                memory,
                direct: projection.is_direct(),
            },
        })
    }

    /// [`stage_memory`](Memory::stage_memory) before a stage of words is read as words.
    fn stage_memory_as_served<S: Numeric>(
        operand: &Tile,
        #[comptime] level: &Level,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] owner: StageOwner,
        #[comptime] element: StageElement,
        #[comptime] decodes: bool,
    ) -> Tile {
        match element {
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
                    // whole along its innermost dimension, or a step would start mid-line.
                    None if decodes => {
                        let innermost = space.dimension_at(space.rank() - 1);
                        space
                            .leaf(operand.place.below_level(level))
                            .whole_run(innermost, source_width)
                    }
                    // A stage laid along a run holds one value a line, of every source line's.
                    None => storage.order().line_width(source_width),
                });
                // A TMA-filled stage's buffer must be TMA-aligned, a gathered one's too: a box of
                // words a gather reads lands as the compacted window the gather stages.
                let delivery = operand.delivery();
                let alignment = comptime!(delivery.engine().stage_alignment);
                // A decoded stage is dense over the values' dimensions whatever the source's map,
                // and so is a tiled one, which an instruction reads as one matrix.
                let tiled = comptime!(matches!(storage, StageStorage::Tiled { .. }));
                if comptime!(projection.is_direct() || decodes || tiled) {
                    Memory::smem_owned::<S>(&space, vector_size, storage, units, alignment, owner)
                } else {
                    owner.refuse_for_a_plane("gathered");
                    Memory::smem_gathered::<S>(
                        &space,
                        vector_size,
                        storage,
                        units,
                        projection,
                        &operand.runtime_map(),
                        operand.window_signed(),
                        operand.window_boundaries(),
                        alignment,
                        owner,
                    )
                }
            }
            StageElement::Stored => {
                owner.refuse_for_a_plane("kept in its stored form");
                Memory::smem_stored::<S>(operand, level, storage, owner)
            }
        }
    }

    /// [`stage`](Memory::stage) in the element the operand is stored in, for the cube `owner`
    /// names, whose units fill it.
    fn smem_stored<S: Numeric>(
        operand: &Tile,
        #[comptime] level: &Level,
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
            Packing::Plain | Packing::Words { .. } => {
                // The values the bulk engine lands come packed; a plain stored stage keeps the
                // copy's alignment.
                Memory::smem_owned::<S>(&space, vector_size, storage, units, 0usize, owner)
            }
            packing => Memory::smem_packed::<S>(
                &space,
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

#[cfg(test)]
mod tests {
    use crate::test_scope::test_scope;
    use crate::*;

    use super::StageElement;

    const M: Dimension = Dimension(0);
    const K: Dimension = Dimension(1);

    /// The facts the Lowering decides the order of a plain `[m, k]` `f32` operand's stage by,
    /// its caller stating `contracted` and the stage served `width` wide where stated.
    fn factor_stage(contracted: Option<Dimension>, width: Option<usize>) -> FactorStage {
        let scope = test_scope();
        let space = Space::new(&[(M, 16), (K, 16)]);
        let operand = Tile::__expand_shared::<f32>(&scope, space.clone(), StageStorage::STRIDED);
        Memory::__expand_factor_stage::<f32>(
            &scope,
            &operand,
            &space,
            width,
            StageElement::Served,
            contracted,
        )
    }

    /// A matmul's caller states the Dimension its contraction sums the operand along.
    #[test]
    fn a_factor_stage_holds_the_contraction_its_caller_states() {
        assert_eq!(factor_stage(Some(K), None).contracted, Some(K));
    }

    /// An attention's caller states no contraction for its keys and values.
    #[test]
    fn a_factor_stage_holds_no_contraction_where_its_caller_states_none() {
        assert_eq!(factor_stage(None, None).contracted, None);
    }

    /// A plain operand of memory, served at its own width, is copied as it lies.
    #[test]
    fn a_plain_operand_at_its_own_width_is_copied_as_it_lies() {
        assert!(factor_stage(Some(K), None).source.copied_as_it_lies());
    }

    /// A stage served at a width of its own is not copied as it lies.
    #[test]
    fn a_stage_at_a_width_of_its_own_is_not_copied_as_it_lies() {
        assert!(!factor_stage(Some(K), Some(4)).source.copied_as_it_lies());
    }
}

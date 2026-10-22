//! Shared-memory stages: the [`StageForm`] a stage takes and the `smem*` constructors.

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::*;

/// Who a shared-memory stage belongs to: the whole cube, or each of its planes. It decides how
/// many copies of the stage one cube holds, and which units fill each.
///
/// Read off the levels a walk sits under ([`Stages::smem`]), never stated: a walk below a level
/// that hands each plane a region of its own is that plane's, and so are the stages it fills.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum StageOwner {
    /// One stage for the cube, filled by the `fillers` planes it sets aside to fill its stages,
    /// or by every unit of it where it sets none aside.
    Cube { fillers: usize },
    /// One stage for each of the cube's `planes` planes, laid side by side in one allocation,
    /// each filled by its own plane's units and read by nobody else.
    Plane { planes: usize },
}

impl StageOwner {
    /// One stage for the cube, filled by every unit of it.
    pub(crate) const CUBE: Self = StageOwner::Cube { fillers: 0 };

    /// The owner of a stage run by `scope` in a cube of `planes` planes, `fillers` of which the
    /// walk sets aside to fill its stages.
    ///
    /// # Panics
    ///
    /// A unit's scope: a stage is filled cooperatively, and one unit has nobody to fill it with.
    pub(crate) fn new(scope: ComputeScope, planes: usize, fillers: usize) -> Self {
        match scope {
            ComputeScope::Cube => StageOwner::Cube { fillers },
            ComputeScope::Plane => StageOwner::Plane { planes },
            ComputeScope::PlaneGroup { .. } => panic!(
                "StageOwner: this walk hands each plane group a region of its own, and a stage \
                 is not yet laid out per plane group: stage it from the cube above the groups"
            ),
            ComputeScope::Unit => panic!(
                "StageOwner: this walk hands each unit a region of its own, and a stage is \
                 filled cooperatively: stage it from the plane or the cube above the units"
            ),
        }
    }

    /// The units that fill one copy of the stage, where the cube holds `cube_units` (`0` when
    /// unknown): all of them, the filling planes', or one plane's share. A share that is not a
    /// whole count is unknown, which a fill walks rolled.
    pub(crate) fn fill(self, cube_units: usize) -> FillUnits {
        match self {
            StageOwner::Cube { fillers: 0 } => FillUnits::cube(cube_units),
            StageOwner::Cube { fillers } => FillUnits::filling_planes(fillers),
            StageOwner::Plane { planes } => FillUnits {
                scope: ComputeScope::Plane,
                count: match cube_units.is_multiple_of(planes) {
                    true => cube_units / planes,
                    false => 0,
                },
                fillers: 0,
            },
        }
    }
}

#[cube]
impl Memory {
    /// Allocate a shared-memory tile over `space` at physical `vector_size`.
    /// `units` is the launch's cube size, `0` when unknown.
    pub(crate) fn smem<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
    ) -> Tile {
        Memory::smem_owned::<T>(
            space,
            vector_size,
            storage,
            units,
            comptime!(0usize),
            comptime!(StageOwner::CUBE),
        )
    }

    /// [`smem`](Memory::smem) for the stage `owner` holds: the cube's one, or the calling plane's
    /// own copy, with a minimum byte alignment on the buffer, never below one
    /// [`RowChunks::CHUNK_BYTES`] chunk: an `ldmatrix` row address needs it. `units` is the
    /// launch's cube size, `0` when unknown.
    pub(crate) fn smem_owned<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] alignment: usize,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let alignment = comptime!(alignment.max(RowChunks::CHUNK_BYTES));
        let elem_bytes = T::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            vector_size,
            storage,
            LineBytes(vector_size * elem_bytes)
        ));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        Memory::smem_with_form::<T>(
            space,
            vector_size,
            units,
            form,
            map,
            ComptimeOption::new_None(),
            alignment,
            owner,
        )
    }

    /// [`smem`](Memory::smem) for a gathered operand: the compacted physical window its sub-tile
    /// reads ([`Compaction`]), still addressed by the operand's own [`Projection`], filled by the
    /// units of the cube `owner` names.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn smem_gathered<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] projection: Projection,
        map: &RuntimeMap,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let form = comptime!(StageForm::gathered(
            &space,
            vector_size,
            storage,
            &projection
        ));
        let stage_map = RuntimeMap {
            coefficients: map.coefficients.clone(),
            residues: Coords::constant(comptime!(vec![0; form.projection.physical_rank()])),
        };
        let stage_map =
            if comptime!(form.projection.is_rational() || form.projection.has_dynamic_scales()) {
                stage_map.stored()
            } else {
                stage_map
            };
        let source = Memory::pending_source_window(
            comptime!(form.steps.clone()),
            comptime!(signed),
            comptime!(boundaries),
        );
        Memory::smem_with_form::<T>(
            space,
            vector_size,
            units,
            form,
            stage_map,
            ComptimeOption::new_Some(source),
            comptime!(0usize),
            comptime!(owner),
        )
    }

    /// The body every plain smem constructor shares; `alignment` `0` is the element's own.
    ///
    /// A stage each plane owns is one allocation of a copy per plane, the calling plane's copy
    /// windowed out by its position on the launch's `y` ([`Partitioning::cube_dim`]), each copy
    /// starting on the whole buffer's alignment.
    #[allow(clippy::too_many_arguments)]
    fn smem_with_form<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] units: usize,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
        #[comptime] alignment: usize,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let size!(W) = vector_size;
        let cells = comptime!(form.cells());
        let smem = match comptime!(owner) {
            StageOwner::Cube { .. } => {
                if comptime!(alignment > 0) {
                    Shared::<[Vector<T, W>]>::new_aligned_slice(cells, alignment)
                } else {
                    Shared::<[Vector<T, W>]>::new_slice(cells)
                }
            }
            StageOwner::Plane { planes } => {
                let line_bytes = comptime!(vector_size * T::size().comptime());
                let copy = comptime!(aligned_cells(cells, line_bytes, alignment));
                let start = UNIT_POS_Y as usize * copy;
                let end = start + cells;
                if comptime!(alignment > 0) {
                    Shared::<[Vector<T, W>]>::new_aligned_slice(comptime!(copy * planes), alignment)
                        .map(|all| &all[start..end])
                } else {
                    Shared::<[Vector<T, W>]>::new_slice(comptime!(copy * planes))
                        .map(|all| &all[start..end])
                }
            }
        };
        Memory::smem_over::<T, _, _>(
            space,
            vector_size,
            comptime!(owner.fill(units)),
            &smem,
            comptime!(Packing::Plain),
            form,
            map,
            source,
        )
    }

    /// [`smem`](Memory::smem) over the words a [`packed`](Packing::Packed) operand is stored in.
    ///
    /// The buffer starts on `alignment` bytes, never below one [`RowChunks::CHUNK_BYTES`] chunk: an
    /// `ldmatrix` row address needs it, and a TMA box the descriptor's swizzle span.
    pub(crate) fn smem_packed<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
        #[comptime] packing: Packing,
        #[comptime] alignment: usize,
        #[comptime] owner: StageOwner,
    ) -> Tile {
        let alignment = comptime!(alignment.max(RowChunks::CHUNK_BYTES));
        let word_bytes = u32::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            vector_size,
            storage,
            LineBytes(packing.physical(vector_size) * word_bytes)
        ));
        let size!(WP) = comptime!(packing.physical(vector_size));
        let smem =
            Shared::<[Vector<u32, WP>]>::new_aligned_slice(comptime!(form.cells()), alignment);
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        Memory::smem_over::<T, _, _>(
            space,
            vector_size,
            comptime!(owner.fill(units)),
            &smem,
            comptime!(packing),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// The body every smem constructor shares: scalar-erases `smem` and windows the whole buffer,
    /// filled by the units `fill` names.
    #[allow(clippy::too_many_arguments)]
    fn smem_over<T: Numeric, H: Numeric, S: CubePrimitive<Scalar = H>>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] fill: FillUnits,
        smem: &Shared<[S]>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile {
        // Held at the element it is declared at: `u32` words under a packing.
        let backing = Backing::new_Buffer(Raw::<Slice>::of::<H>(unsafe {
            smem.inner_ref()
                .downcast_unchecked::<H>()
                .as_boxed_unchecked()
        }));
        Memory::smem_backed::<T>(
            space,
            vector_size,
            fill,
            backing,
            packing,
            form,
            map,
            source,
            comptime!(Write::Replace),
            comptime!(Overhang::Never),
        )
    }

    /// The whole-buffer window over a shared-memory `backing` laid out as `form`, a write past
    /// its extent masked where `overhang` says a region larger than the buffer writes into it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn smem_backed<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] fill: FillUnits,
        backing: Backing,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
        #[comptime] write: Write,
        #[comptime] overhang: Overhang,
    ) -> Tile {
        let (physical_shape, physical_strides) = storage_layout(comptime!(form.clone()));
        let (origin, extent) = full_window(comptime!(form.clone()));
        // The bound is the buffer's extent: a region past it is one a masked write skips.
        let bound = extent.clone();
        let gmem_projection = comptime!(form.positional.clone());
        Tile {
            kind: TileKind::new_Memory(Memory {
                address: comptime!(AddressSpace::Shared),
                store: Store::untiled(backing, vector_size, packing),
                layout: BufferLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                    rows: comptime!(form.rows),
                    paging: ComptimeOption::new_None(),
                },
                // Empty: a list over the buffer's own dims would misalign one level down.
                window: Window::new(origin, extent, bound, false, comptime!(SmallVec::new())),
                projection: comptime!(form.projection),
                map,
                offsets: Coords::<i32>::new(),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang,
                    write,
                    fill,
                    storage: WindowStorage::Stored(Storage::Strided),
                    // Read as a source, a stage is copied out by the units that read it.
                    delivery: Delivery::SyncPerUnit,
                }),
                unit_share: comptime!(UnitShare::Repeated),
                split_share: comptime!(SplitShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                contraction: comptime!(None),
                codebook: Codebook::none(),
                index: ComptimeOption::new_None(),
                source_window: source,
            }),
            factor: Factor::none(),
            place: comptime!(Placement::new(space, 0usize, Vec::new())),
            element: elem_type_of::<T>(),
        }
    }

    /// One plane's landing: a dense scalar stage over `space`, one window per plane in one buffer.
    /// The buffer is returned beside the tile.
    pub(crate) fn landing<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] units: usize,
        #[comptime] planes: usize,
    ) -> (Tile, Shared<[T]>) {
        let cells = comptime!(
            (0..space.rank())
                .map(|p| space.extent_at(p))
                .product::<usize>()
        );
        let start = PLANE_POS.cast::<usize>() * cells;
        let end = start + cells;
        let window =
            Shared::<[T]>::new_slice(comptime!(cells * planes)).map(|all| &all[start..end]);
        let elem_bytes = T::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(elem_bytes)
        ));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let tile = Memory::smem_over::<T, _, _>(
            space,
            1usize,
            comptime!(FillUnits::cube(units)),
            &window,
            comptime!(Packing::Plain),
            form,
            map,
            ComptimeOption::new_None(),
        );
        (tile, window)
    }

    /// An unfilled [`SourceWindow`] for a gathered stage; each [`fill_from`](Memory::fill_from)
    /// writes its origin and bound.
    fn pending_source_window(
        #[comptime] steps: SmallVec<[usize; Space::MAX_RANK]>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    ) -> SourceWindow {
        let rank = comptime!(steps.len());
        let mut origin = Coords::<i32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for _ in 0..rank {
            origin.push(0i32);
            bound.push(0u32);
        }
        SourceWindow {
            origin,
            bound,
            steps: comptime!(steps),
            signed: comptime!(signed),
            boundaries: comptime!(boundaries),
        }
    }
}

/// The whole-buffer window of a stage: zero origin, its extents along the axes its buffer is
/// addressed by ([`window_extents`](StageForm::window_extents)).
#[cube]
#[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
fn full_window(#[comptime] form: StageForm) -> (Coords<i32>, Coords<u32>) {
    let mut origin = Coords::<i32>::new();
    let mut extent = Coords::<u32>::new();
    let extents = comptime!(form.window_extents());

    #[unroll]
    for p in 0..comptime!(extents.len()) {
        origin.push(0);
        extent.push(comptime!(extents[p] as u32).runtime());
    }

    (origin, extent)
}

/// The [`Compaction`] of a gathered source's map, which `dst` must be addressed by; `None` when
/// neither side gathers. Panics if `dst` is not that compaction's projection.
pub(crate) fn stage_compaction(
    src: &Projection,
    dst: &Projection,
    vector_size: usize,
    space: &Space,
) -> Option<Compaction> {
    // Direct in coordinate space means no gather, tiled buffers included.
    if src.is_direct() && dst.is_direct() {
        return None;
    }
    // A partition stages like a direct operand: nothing aliases.
    if src.composition() == Composition::Disjoint && dst.is_direct() {
        return None;
    }
    let compaction = Compaction::new(src, vector_size, |axis| space.extent(axis));
    assert!(
        compaction.projection() == dst,
        "stage_compaction: a gathered source fills the compacted stage of its own \
         projection, addressed by {:?}, but the destination is addressed by {dst:?}",
        compaction.projection()
    );
    Some(compaction)
}

/// A stage's buffer: its physical extents and the two mappings that address them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct StageForm {
    /// Physical extents in lines.
    extents: Vec<usize>,
    /// The buffer's own per-position map.
    positional: Projection,
    /// How the staged tile's logical axes address those extents.
    projection: Projection,
    /// Per physical axis, what a stage coordinate is multiplied by to land on the source.
    steps: SmallVec<[usize; Space::MAX_RANK]>,
    /// Where each line of a block row is kept.
    pub(crate) rows: RowArrangement,
}

impl StageForm {
    /// A dense copy of the logical tile, one `[grid…, block…]` split per `nesting` block.
    pub(crate) fn dense(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        line: LineBytes,
    ) -> StageForm {
        let nesting = stage.nesting(space);
        let extents = StageForm::dense_extents(space, vector_size, &nesting);
        let rows = match stage {
            StageStorage::Tiled { chunks, .. } => RowArrangement::new(chunks, &extents, line),
            StageStorage::Strided | StageStorage::Lines { .. } => RowArrangement::InOrder,
        };
        StageForm {
            extents,
            rows,
            positional: StageForm::positional(space.rank(), nesting.len() + 1),
            // A dense stage addresses its own buffer directly, whatever its operand was gathered
            // through.
            projection: Projection::direct_over(space),
            steps: SmallVec::new(),
        }
    }

    /// The compacted row-major window a gathered operand stages into ([`Compaction`]).
    fn gathered(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        projection: &Projection,
    ) -> StageForm {
        assert!(
            matches!(stage, StageStorage::Strided),
            "StageForm: a gathered operand stages into a plain row-major window, but {stage:?} \
             storage was asked for"
        );
        let compaction = Compaction::new(projection, vector_size, |axis| space.extent(axis));
        let extents = compaction.line_extents(vector_size);
        StageForm {
            rows: RowArrangement::InOrder,
            positional: StageForm::positional(extents.len(), 1),
            projection: compaction.projection().clone(),
            steps: compaction.steps().iter().copied().collect(),
            extents,
        }
    }

    /// The extent in lines along each axis the buffer is addressed by, a window's axes: what the
    /// physical extents carrying that axis hold between them, so a buffer tiled into
    /// `[grid…, block…]` is still windowed one coordinate an axis.
    fn window_extents(&self) -> Vec<usize> {
        (0..self.positional.coordinate_rank())
            .map(|c| {
                self.positional
                    .carriers(Axis(c as u8))
                    .iter()
                    .map(|&p| self.extents[p])
                    .product()
            })
            .collect()
    }

    /// The buffer's rank: how many physical axes its layout addresses.
    pub(crate) fn physical_rank(&self) -> usize {
        self.projection.physical_rank()
    }

    pub(crate) fn cells(&self) -> usize {
        self.pitched().iter().product()
    }

    /// Row-major strides over [`extents`](StageForm::extents), rows pitched past their padding.
    fn strides(&self) -> Vec<usize> {
        let pitched = self.pitched();
        (0..pitched.len())
            .map(|p| pitched[p + 1..].iter().product())
            .collect()
    }

    /// [`extents`](StageForm::extents) with each row widened by its padding lines.
    fn pitched(&self) -> Vec<usize> {
        let mut pitched = self.extents.clone();
        if let Some(last) = pitched.last_mut() {
            *last += self.rows.padding();
        }
        pitched
    }

    /// A stage's buffer addressed by position: `rank` synthetic axes, each split into `pieces` dims.
    fn positional(rank: usize, pieces: usize) -> Projection {
        let axes: Vec<Axis> = (0..rank).map(|p| Axis(p as u8)).collect();
        let labels = StoragePartitioning::level_major(&axes, &vec![pieces; rank]);
        Projection::tiled(&axes, &labels)
    }

    /// A dense stage's physical line extents: `[extents…]` or `[grid…, …, block…]`.
    fn dense_extents(space: &Space, vector_size: usize, nesting: &[Space]) -> Vec<usize> {
        let rank = space.rank();
        let mut extents = Vec::new();
        let mut outer = space;
        for block in nesting {
            for p in 0..rank {
                let (e, b) = (outer.extent_at(p), block.extent_at(p));
                assert!(
                    e.is_multiple_of(b),
                    "Memory::smem: a {b}-element storage block must divide the {e}-element block \
                     enclosing it on axis {p}"
                );
                extents.push(e / b);
            }
            outer = block;
        }
        for p in 0..rank {
            extents.push(outer.extent_at(p));
        }
        // Rounded up: the last line's spare units are padding.
        let last = extents.len() - 1;
        extents[last] = extents[last].div_ceil(vector_size);
        extents
    }
}

/// A stage's physical shape and strides, in lines.
#[cube]
fn storage_layout(#[comptime] form: StageForm) -> (Coords<u32>, Coords<u32>) {
    let strides_c = comptime!(form.strides());

    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for p in 0..comptime!(form.extents.len()) {
        shape.push(comptime!(form.extents[p] as u32));
        strides.push(comptime!(strides_c[p] as u32));
    }

    (shape, strides)
}

/// Lines one plane's copy of a stage of `cells` lines, each `line_bytes` long, spans in a buffer
/// of one copy per plane: rounded up so the next copy starts on `alignment` bytes (`0` = the
/// element's own, which every line already keeps).
fn aligned_cells(cells: usize, line_bytes: usize, alignment: usize) -> usize {
    match alignment > line_bytes && alignment.is_multiple_of(line_bytes) {
        true => cells.next_multiple_of(alignment / line_bytes),
        false => cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// `16 -> 8 -> 4` on both axes: three distinct block shapes.
    fn space() -> (Space, Vec<Level>) {
        (
            Space::new(&[(M, 16), (N, 16)]),
            Levels::leaf(&[(M, 4), (N, 4)])
                .walk(&[(M, 2), (N, 2)])
                .walk_every(&[M, N])
                .build(),
        )
    }

    /// [`space`] plus an ungathered innermost axis, cut to `8 x 8 x 4`.
    fn gathered_space() -> Space {
        Space::new(&[(M, 8), (N, 8), (K, 4)])
    }

    /// No nesting is the plain row-major buffer.
    #[test]
    fn flat_nesting_is_the_space_itself() {
        let (space, _) = space();
        assert_eq!(StageForm::dense_extents(&space, 1, &[]), vec![16, 16]);
        assert_eq!(StageForm::dense_extents(&space, 4, &[]), vec![16, 4]);
    }

    /// One block is the `[grid…, tile…]` split.
    #[test]
    fn one_block_splits_grid_and_tile() {
        let (space, levels) = space();
        assert_eq!(
            StageForm::dense_extents(&space, 1, &[space.leaf(&levels)]),
            vec![4, 4, 4, 4]
        );
    }

    /// Two nested blocks add a middle grid: `16 = 2 x (8 = 2 x (4))`.
    #[test]
    fn two_blocks_nest() {
        let (space, levels) = space();
        let nesting = [levels[0].child(&space), space.leaf(&levels)];
        let extents = StageForm::dense_extents(&space, 1, &nesting);
        assert_eq!(extents, vec![2, 2, 2, 2, 4, 4]);
        assert_eq!(extents.iter().product::<usize>(), space.cells());
    }

    /// A buffer's extents imply row-major strides.
    #[test]
    fn a_form_strides_row_major() {
        let (space, levels) = space();
        let form = StageForm::dense(&space, 4, StageStorage::Strided, LineBytes(16));
        assert_eq!(form.extents, vec![16, 4]);
        assert_eq!(form.strides(), vec![4, 1]);
        assert_eq!(form.cells(), space.cells() / 4);

        let tiled = StageForm::dense(
            &space,
            1,
            StageStorage::Tiled {
                block: space.leaf(&levels).extents(),
                chunks: RowChunks::InOrder,
            },
            LineBytes(4),
        );
        assert_eq!(tiled.extents, vec![4, 4, 4, 4]);
        assert_eq!(tiled.strides(), vec![64, 16, 4, 1]);
    }

    /// A stage's bytes are its lines, and a padded block's rows each a chunk longer: `16 × 16`
    /// `f16` values in 8-byte lines are 512 bytes, and 768 with 16 bytes after each of the 16 rows.
    #[test]
    fn a_stage_holds_its_lines_and_its_padding() {
        let (space, _) = space();
        assert_eq!(StageStorage::Strided.bytes(&space, 4, 16), 512);
        let padded = StageStorage::Tiled {
            block: space.extents(),
            chunks: RowChunks::Padded,
        };
        assert_eq!(padded.bytes(&space, 4, 16), 768);
    }

    /// A swizzled block keeps in-order extents and strides, swizzling only 16-byte lines.
    #[test]
    fn a_swizzled_block_is_laid_out_as_one_in_order() {
        let (space, levels) = space();
        let swizzled = |line_bytes| {
            let line = LineBytes(line_bytes);
            StageForm::dense(
                &space,
                1,
                StageStorage::Tiled {
                    block: space.leaf(&levels).extents(),
                    chunks: RowChunks::Swizzled,
                },
                line,
            )
        };
        assert_eq!(swizzled(4).extents, vec![4, 4, 4, 4]);
        assert_eq!(swizzled(4).strides(), vec![64, 16, 4, 1]);
        assert_eq!(swizzled(4).rows, RowArrangement::InOrder);
        let RowArrangement::Swizzled(swizzle) = swizzled(16).rows else {
            panic!("rows of four 16-byte lines are swizzled")
        };
        assert_eq!((swizzle.row_axis(), swizzle.line_axis()), (2, 3));
    }

    /// One box lands a stage whose blocks stack whole rows, swizzled as the stage keeps them by
    /// the engine's span for the row's bytes; blocks splitting a row, or padded rows, it cannot.
    #[test]
    fn a_box_lands_a_stage_of_whole_rows_swizzled_by_their_span() {
        let tiled = |k_block, chunks| StageStorage::Tiled {
            block: vec![(M, 16), (K, k_block)],
            chunks,
        };
        let stage = |k| Space::new(&[(M, 64), (K, k)]);
        let f16_bits = 16;
        for (k, mode) in [
            (16, TensorMapSwizzle::B32),
            (32, TensorMapSwizzle::B64),
            (64, TensorMapSwizzle::B128),
        ] {
            assert_eq!(
                tiled(k, RowChunks::Swizzled).tma_swizzle(&stage(k), 1, f16_bits),
                Ok(mode)
            );
        }
        assert_eq!(
            tiled(32, RowChunks::InOrder).tma_swizzle(&stage(32), 1, f16_bits),
            Ok(TensorMapSwizzle::None)
        );
        assert_eq!(
            StageStorage::Strided.tma_swizzle(&stage(32), 8, f16_bits),
            Ok(TensorMapSwizzle::None)
        );
        assert!(matches!(
            tiled(16, RowChunks::Swizzled).tma_swizzle(&stage(32), 1, f16_bits),
            Err(Refusal::BoxSplitsStageBlocks { axis: K, .. })
        ));
        assert!(matches!(
            tiled(32, RowChunks::Padded).tma_swizzle(&stage(32), 1, f16_bits),
            Err(Refusal::RowsNoDescriptorLands { .. })
        ));
        assert!(matches!(
            tiled(128, RowChunks::Swizzled).tma_swizzle(&stage(128), 1, f16_bits),
            Err(Refusal::RowsNoDescriptorLands { .. })
        ));
    }

    /// A swizzled box keys its rows off their place in the stage, so a block of rows the swizzle
    /// does not repeat over lands its later blocks under the wrong keys.
    #[test]
    fn a_swizzled_box_refuses_blocks_off_the_swizzles_period() {
        let storage = StageStorage::Tiled {
            block: vec![(M, 4), (K, 32)],
            chunks: RowChunks::Swizzled,
        };
        assert_eq!(
            storage.tma_swizzle(&Space::new(&[(M, 64), (K, 32)]), 1, 16),
            Err(Refusal::BoxSplitsStageBlocks {
                axis: K,
                rows: 4,
                period: 8
            })
        );
    }

    /// A padded block pitches its rows one chunk further apart.
    #[test]
    fn a_padded_block_pitches_its_rows_past_a_chunk() {
        let (space, levels) = space();
        let padded = StageForm::dense(
            &space,
            1,
            StageStorage::Tiled {
                block: space.leaf(&levels).extents(),
                chunks: RowChunks::Padded,
            },
            LineBytes(16),
        );
        assert_eq!(padded.extents, vec![4, 4, 4, 4]);
        assert_eq!(padded.rows, RowArrangement::Padded { lines: 1 });
        assert_eq!(padded.strides(), vec![80, 20, 5, 1]);
        assert_eq!(padded.cells(), 320);
    }

    /// A gathered stage is the compacted window, not the logical tile.
    #[test]
    fn a_gathered_form_is_the_compacted_window() {
        let space = gathered_space();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        let form = StageForm::gathered(&space, 1, StageStorage::Strided, &projection);
        // 8 x 8 logical cells over 1 + 7 + 7 physical ones, times 4 of `K`.
        assert_eq!(form.extents, vec![15, 4]);
        assert_eq!(form.cells(), 60);
        assert_eq!(form.projection, projection);
        assert!(form.positional.is_direct());
    }

    /// A gathered operand's stage refuses tiled storage.
    #[test]
    #[should_panic(expected = "plain row-major window")]
    fn a_gathered_form_refuses_tiled_storage() {
        let space = gathered_space();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        StageForm::gathered(
            &space,
            1,
            StageStorage::Tiled {
                block: vec![(M, 4), (N, 4), (K, 4)],
                chunks: RowChunks::InOrder,
            },
            &projection,
        );
    }

    /// A block must divide the one enclosing it.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_block_must_divide_its_enclosing_block() {
        let (space, levels) = space();
        // Reversed: the coarse block sits inside the fine one.
        StageForm::dense_extents(&space, 1, &[space.leaf(&levels), levels[0].child(&space)]);
    }

    /// A `Tiled` stage groups the stated block, unless the space already is it.
    #[test]
    fn the_nesting_follows_the_layout() {
        let (space, levels) = space();
        let tiled = StageStorage::Tiled {
            block: space.leaf(&levels).extents(),
            chunks: RowChunks::InOrder,
        };
        assert!(tiled.nesting(&space)[0] == space.leaf(&levels));
        assert!(StageStorage::Strided.nesting(&space).is_empty());
        assert!(tiled.nesting(&space.leaf(&levels)).is_empty());
    }

    /// A plane fills its own copy with its share of the cube's units, and a share that is not a
    /// whole count is read as unknown rather than rounded.
    #[test]
    fn a_plane_fills_with_its_share_of_the_units() {
        let plane = StageOwner::Plane { planes: 4 };
        assert_eq!(plane.fill(128).count, 32);
        assert_eq!(plane.fill(0).count, 0);
        assert_eq!(plane.fill(130).count, 0);
        assert_eq!(plane.fill(128).scope, ComputeScope::Plane);
        assert_eq!(StageOwner::CUBE.fill(128), FillUnits::cube(128));
    }

    /// A cube's stage under planes set aside is filled by their units alone, the planes at the
    /// end of the cube, so the planes contracting never take a share of its fill.
    #[test]
    fn a_cube_setting_planes_aside_fills_with_their_units() {
        let fill = StageOwner::Cube { fillers: 1 }.fill(160);
        assert_eq!(fill, FillUnits::filling_planes(1));
        assert_eq!(fill.scope, ComputeScope::Cube);
        assert_eq!(fill.count, 0);
    }

    /// Every plane's copy starts on the buffer's alignment: a copy of lines shorter than it is
    /// rounded up to whole alignments, and a line already as long keeps its count.
    #[test]
    fn every_planes_copy_starts_aligned() {
        assert_eq!(aligned_cells(5, 4, 16), 8);
        assert_eq!(aligned_cells(8, 4, 16), 8);
        assert_eq!(aligned_cells(5, 16, 16), 5);
        assert_eq!(aligned_cells(5, 32, 16), 5);
        assert_eq!(aligned_cells(5, 4, 0), 5);
    }
}

//! A table of `u32` entries an operand carries ([`TileArgLaunch::indexed`]), read where its window
//! is cut: each row's expert, each tile's group, each group's run of rows.
//!
//! The table's dims are its key axes, in order: it holds one entry per tile of the first level of
//! the nest that cuts them, the same entry a kernel would read off `region.coord(key)` itself.
//! The tile reads that entry once, in [`Tile::at`], at the step that cuts the keys, and places its
//! window with it ([`Indexing`]); its reads never touch the table again. Where no level cuts the
//! keys they are one tile, so the table's first entry is the one, read as the tile is served. The scales it carries move
//! with it wherever they span the axis the entry places, so a routed operand may be quantized.

use cubecl::prelude::*;

use crate::stage::pipeline::payload::base::StageOperand;
use crate::*;

/// What an operand's table entry places. The table's own dims are its keys
/// ([`TileArgLaunch::indexed`]).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Indexing {
    /// The entry is the operand's coordinate along `axis`, which no level may cut: each row's
    /// expert, keyed by the row; each tile's group, keyed by the row axis a level cuts in tiles.
    /// The axis is one wide below it, so a contraction sees one slab. An entry past the axis reads
    /// the last coordinate, unless the operand checks the axis with [`Boundary::Zero`], where it
    /// reads zero.
    Coordinate(Axis),
    /// The entry is where the operand's run along `axis` ends, and the one before it along the
    /// key where it starts (`0` for the first): each group's rows packed end to end, the table
    /// their cumulative ends. The operand reads zero past the run, and a walk over its window
    /// along `axis` steps the run alone where the axis is dynamic. A run that ends before it
    /// starts is empty. No level at or above the key's may cut `axis`.
    Run(Axis),
}

impl Indexing {
    /// The axis the entry places.
    pub fn axis(&self) -> Axis {
        match self {
            Indexing::Coordinate(axis) | Indexing::Run(axis) => *axis,
        }
    }

    /// What an operand of `values` owes an index keyed by `keys`, checked as the table is bound.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first rule the statement breaks.
    pub(crate) fn validate(
        &self,
        values: &TileSpec,
        width: usize,
        keys: &[Axis],
    ) -> Result<(), Refusal> {
        let axis = self.axis();
        if !values.axes().contains(&axis) {
            return Err(Refusal::IndexOfAnAxisNotSpanned(axis));
        }
        if keys.is_empty() || keys.contains(&axis) {
            return Err(Refusal::IndexKeyedByItsAxis(axis));
        }
        if matches!(self, Indexing::Run(_)) && keys.len() != 1 {
            return Err(Refusal::RunKeyedByMany(keys.len()));
        }
        self.places(values, width)
    }

    /// What an operand bound `width` values a line owes where an entry moves its window, the
    /// values' or a scale's spanning the placed axis: a buffer neither storage-tiled nor paged, one
    /// dim of which is the axis alone, read one value a line where that dim is the innermost, since
    /// an entry is an element and a line would straddle two; under a run, a buffer over its
    /// labelled dims, as [`Tile::within`] places.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first rule `spec` breaks.
    pub(crate) fn places(&self, spec: &TileSpec, width: usize) -> Result<(), Refusal> {
        let projection = &spec.projection;
        let stated =
            projection.is_tiled() || !spec.stored_tiles.is_empty() || spec.paging.is_some();
        let gathered = matches!(self, Indexing::Run(_)) && !projection.is_direct();
        let dim = placed_dim(projection, self.axis());
        match dim {
            _ if stated || gathered => Err(Refusal::IndexOfAStatedLayout),
            None => Err(Refusal::IndexOfAStatedLayout),
            Some(dim)
                if dim + 1 == projection.physical_rank()
                    && (width > 1 || spec.packing != Packing::Plain) =>
            {
                Err(Refusal::IndexAlongALine(self.axis()))
            }
            Some(_) => Ok(()),
        }
    }

    /// Where in `levels` the table is read: the first level that cuts a key, which must cut them
    /// all, and below which (for a run, at or above which) no level cuts the placed axis; `None`
    /// where no level cuts the keys, which are then one tile, read as the tile is served.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first rule the nest breaks.
    pub(crate) fn read_at(
        &self,
        keys: &[Axis],
        levels: &[Level],
    ) -> Result<Option<usize>, Refusal> {
        let cuts = |level: &Level, axis: Axis| level.count(axis).is_some();
        let at = levels
            .iter()
            .position(|level| keys.iter().any(|&key| cuts(level, key)));
        if let Some(at) = at
            && !keys.iter().all(|&key| cuts(&levels[at], key))
        {
            return Err(Refusal::IndexKeysCutApart(keys.to_vec()));
        }
        let axis = self.axis();
        let placed_by_a_level = match (self, at) {
            (Indexing::Coordinate(_), _) => levels.iter().any(|level| cuts(level, axis)),
            (Indexing::Run(_), Some(at)) => levels[..=at].iter().any(|level| cuts(level, axis)),
            (Indexing::Run(_), None) => false,
        };
        match placed_by_a_level {
            true => Err(Refusal::IndexedAxisCut(axis)),
            false => Ok(at),
        }
    }
}

/// The buffer dim that is `axis` alone, which an entry moves the window along.
pub(crate) fn placed_dim(projection: &Projection, axis: Axis) -> Option<usize> {
    Witness::new(projection, axis).map(|witness| witness.dim())
}

/// A table as a tile reads it: its entries, its stride along each key, and whether its entry is
/// placed yet. Held by the values' [`Memory`], beside the window it places.
// Read through its expand alone: `Tile::indexed_at` is written at expand time.
#[allow(dead_code)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct IndexTable {
    entries: Box<[u32]>,
    /// The table's stride along each key, in entries.
    strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) indexing: Indexing,
    /// The table's dims, in order.
    #[cube(comptime)]
    pub(crate) keys: Vec<Axis>,
    /// Whether the entry is read and the window placed: a placed table is never read again.
    #[cube(comptime)]
    pub(crate) placed: bool,
    /// Whether the entry read names a coordinate of the axis, or a run of at least one; `true`
    /// until it is read ([`Tile::entry_in_range`]).
    in_range: bool,
}

#[cube]
impl IndexTable {
    /// `table`, its dims the `keys`, placing as `indexing` says.
    #[allow(dead_code)] // Reached through its expand, from `TileArg::tile`.
    pub(crate) fn new(
        table: &Tensor<u32>,
        #[comptime] keys: Vec<Axis>,
        #[comptime] indexing: Indexing,
    ) -> IndexTable {
        let mut strides = Coords::<u32>::new();
        #[unroll]
        for i in 0..comptime!(keys.len()) {
            strides.push(table.stride(i) as u32);
        }
        IndexTable {
            entries: unsafe { table.as_slice().as_boxed_unchecked() },
            strides,
            indexing,
            keys,
            placed: comptime!(false),
            in_range: true,
        }
    }

    /// The run whose end is the entry at `at`, `(start, end)`: its start the entry before it
    /// along the one key, or `0` for the first.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn run(&self, at: u32) -> (u32, u32) {
        let stride = self.strides.at(0usize);
        let previous = self.entry(at.max_with(stride).minus(stride));
        (select(at == 0u32, 0u32, previous), self.entry(at))
    }

    /// Where the entry of `step`'s tile of the keys lies in the table.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn position(&self, step: &Step) -> u32 {
        let mut at = 0u32;
        #[unroll]
        for i in 0..comptime!(self.keys.len()) {
            let key = comptime!(self.keys[i]);
            at = at.plus(step.coord(key).cast::<u32>().times(self.strides.at(i)));
        }
        at
    }

    /// The entry at `at`, or the table's last past it: the table is device data a launch sized,
    /// and a read past it would land outside the buffer.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn entry(&self, at: u32) -> u32 {
        let last = (self.entries.len() as u32).max_with(1u32).minus(1u32);
        self.entries[at.min_with(last).cast::<usize>()]
    }

    /// This table with its entry read and placed, `in_range` saying whether the entry named a
    /// coordinate of the axis, or a run of at least one.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn read(&self, in_range: bool) -> IndexTable {
        IndexTable {
            entries: self.entries.clone(),
            strides: self.strides.clone(),
            indexing: comptime!(self.indexing.clone()),
            keys: comptime!(self.keys.clone()),
            placed: comptime!(true),
            in_range,
        }
    }
}

/// Whether the run `[from, until)` holds a position.
#[cube]
fn nonempty(from: u32, until: u32) -> bool {
    from < until
}

/// What a tile with no table to read answers: its window is where it lies.
#[cube]
fn no_table() -> bool {
    true
}

/// Where the first entry lies: the one a table no level's keys cut is read at.
#[cube]
fn first_entry() -> u32 {
    0u32
}

impl Tile {
    /// Whether the entry this tile's table placed it with names a coordinate of the axis it
    /// places, or a run of at least one row: `false` for an entry past a stack of slabs, as a
    /// padding row's route or a padding tile's group names. A kernel skips such a tile's work
    /// rather than read a clamped slab. `true` for a tile with no table, or none read yet.
    pub fn entry_in_range(&self) -> bool {
        cubecl::unexpanded!()
    }

    /// This tile with its table read where `step` cuts the table's keys: its window, and the
    /// windows of the scales it carries that span the axis the entry places, placed by the entry.
    /// Any other tile, or step, as it is.
    pub(crate) fn indexed_at(self, _step: &Step) -> Tile {
        cubecl::unexpanded!()
    }

    /// This tile with `table` to read where a step cuts its keys.
    #[allow(dead_code)] // Reached through its expand, from `TileArg::tile`.
    pub(crate) fn with_index(self, _table: IndexTable) -> Tile {
        cubecl::unexpanded!()
    }

    /// This tile with its table read at its first entry: what a table whose keys no level cuts
    /// holds, the keys being one tile.
    #[allow(dead_code)] // Reached through its expand, from `TileArg::tile`.
    pub(crate) fn indexed_whole(self) -> Tile {
        cubecl::unexpanded!()
    }
}

impl TileExpand {
    pub fn __expand_entry_in_range_method(&self, scope: &Scope) -> NativeExpand<bool> {
        match &self.kind {
            TileKindExpand::Memory(MemoryExpand {
                index: ComptimeOptionExpand::Some(table),
                ..
            }) => table.in_range,
            _ => no_table::expand(scope),
        }
    }

    pub(crate) fn __expand_indexed_at_method(self, scope: &Scope, step: &StepExpand) -> TileExpand {
        let TileKindExpand::Memory(memory) = &self.kind else {
            return self;
        };
        let ComptimeOptionExpand::Some(table) = &memory.index else {
            return self;
        };
        let cuts = |key: &Axis| step.level.count(*key).is_some();
        if table.placed || !table.keys.iter().any(cuts) {
            return self;
        }
        assert!(
            table.keys.iter().all(cuts),
            "{}",
            Refusal::IndexKeysCutApart(table.keys.clone())
        );
        let at = table.__expand_position_method(scope, step);
        self.read_index(scope, at)
    }

    pub(crate) fn __expand_indexed_whole_method(self, scope: &Scope) -> TileExpand {
        let at = first_entry::expand(scope);
        self.read_index(scope, at)
    }

    /// This tile placed by the entry at `at` of the table it holds, which is then read.
    fn read_index(self, scope: &Scope, at: NativeExpand<u32>) -> TileExpand {
        let TileKindExpand::Memory(MemoryExpand {
            index: ComptimeOptionExpand::Some(table),
            ..
        }) = &self.kind
        else {
            unreachable!("only a tile holding a table reads it")
        };
        let table = table.clone();
        let (placed, in_range) = match table.indexing {
            Indexing::Coordinate(axis) => {
                let entry = table.__expand_entry_method(scope, at);
                let in_range = self.__expand_names_coordinate_method(scope, axis, entry);
                let entry = self.__expand_clamped_method(scope, axis, entry);
                (self.__expand_placed_at_method(scope, axis, entry), in_range)
            }
            Indexing::Run(axis) => {
                let (start, end) = table.__expand_run_method(scope, at);
                (
                    self.__expand_run_within_method(scope, axis, start, end),
                    nonempty::expand(scope, start, end),
                )
            }
        };
        let read = table.__expand_read_method(scope, in_range);
        placed.__expand_with_index_method(scope, read)
    }

    pub(crate) fn __expand_with_index_method(
        mut self,
        _scope: &Scope,
        table: IndexTableExpand,
    ) -> TileExpand {
        match &mut self.kind {
            TileKindExpand::Memory(memory) => memory.index = ComptimeOptionExpand::Some(table),
            _ => panic!("{}", Refusal::IndexOfAStatedLayout),
        }
        self
    }
}

#[cube]
impl Tile {
    /// Whether `coord`, an entry naming a coordinate along `axis`, lies within the axis.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    fn names_coordinate(&self, #[comptime] axis: Axis, coord: u32) -> bool {
        let memory = self.mem("indexed_at");
        let p = comptime!(placed_dim(&memory.projection, axis).unwrap());
        coord < memory.window.bound.at(p)
    }

    /// `coord`, an entry naming a coordinate along `axis`, clamped to the axis unless the operand
    /// reads zero past it.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    fn clamped(&self, #[comptime] axis: Axis, coord: u32) -> u32 {
        let memory = self.mem("indexed_at");
        let p = comptime!(placed_dim(&memory.projection, axis).unwrap());
        let zero =
            comptime!(memory.window.boundaries.get(p).copied().flatten() == Some(Boundary::Zero));
        if comptime!(zero) {
            coord
        } else {
            coord.min_with(memory.window.bound.at(p).max_with(1u32).minus(1u32))
        }
    }

    /// This tile one wide at `coord` along `axis`, and so are the scales it carries that span it.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn placed_at(&self, #[comptime] axis: Axis, coord: u32) -> Tile {
        Tile {
            kind: TileKind::new_Memory(self.mem("placed_at").placed_at(axis, coord)),
            factor: self.factor.placed_at(axis, coord),
            place: comptime!(Placement::new(
                Level::every(&[(axis, 1)]).child(&self.place.space),
                self.place.depth,
                self.place.levels.clone()
            )),
            element: comptime!(self.element),
        }
    }

    /// This tile reading the run `[from, until)` along `axis`, zero past it, and so are the scales
    /// it carries that span it. An `until` before `from` is an empty run, never a negative one.
    #[allow(dead_code)] // Reached through its expand, from `Tile::indexed_at`.
    pub(crate) fn run_within(&self, #[comptime] axis: Axis, from: u32, until: u32) -> Tile {
        let until = until.max_with(from);
        Tile {
            kind: TileKind::new_Memory(self.mem("run_within").within(
                axis,
                from.cast::<usize>(),
                until.cast::<usize>(),
            )),
            factor: self.factor.run_within(axis, from, until),
            place: comptime!(self.place.clone()),
            element: comptime!(self.element),
        }
    }
}

impl Region {
    /// The regions of the level below this one, over the runs `tile` holds: along a dynamic axis
    /// an index placed a run of `tile` along ([`Indexing::Run`]), the run's tiles alone; along every
    /// other, this region's box. What a cube walks its group's rows by, every operand it holds
    /// windowed by the same regions.
    pub fn walk_within(&self, _tile: &Tile) -> Walk {
        cubecl::unexpanded!()
    }
}

impl RegionExpand {
    pub fn __expand_walk_within_method(&self, scope: &Scope, tile: &TileExpand) -> WalkExpand {
        let child = self.__expand_child_method(scope);
        let box_space = SpaceExpand::comptime(&child);
        let run = tile
            .run_axis()
            .filter(|&axis| box_space.contains(axis) && box_space.is_dynamic(axis));
        let sized = match run {
            Some(axis) => {
                let extent = tile.clone().__expand_runtime_extent_method(scope, axis);
                with_extent::expand(scope, &child, box_space.position(axis), extent)
            }
            None => child,
        };
        let level = self.path.next();
        let mut walk = Walk::__expand_of(scope, &sized, level.clone(), self.clone());
        walk.run = run.filter(|&axis| level.axes() == [axis] && !level.is_reversed(axis));
        walk
    }
}

/// `space` with the runtime size of its dynamic axis at `p` set to `extent`.
#[cube]
fn with_extent(space: &Space, #[comptime] p: usize, extent: usize) -> Space {
    let mut sizes = Sequence::<usize>::new();
    #[unroll]
    for i in 0..comptime!(space.rank()) {
        if comptime!(i == p) {
            sizes.push(extent);
        } else {
            sizes.push(space.runtime_extent_at(i));
        }
    }
    Space::with_sizes(comptime!(space.clone()), sizes)
}

impl TileExpand {
    /// The axis an index placed a run of this tile along, where it holds one.
    pub(crate) fn run_axis(&self) -> Option<Axis> {
        match &self.kind {
            TileKindExpand::Memory(MemoryExpand {
                index: ComptimeOptionExpand::Some(table),
                ..
            }) if table.placed => match table.indexing {
                Indexing::Run(axis) => Some(axis),
                Indexing::Coordinate(_) => None,
            },
            _ => None,
        }
    }
}

impl Tile {
    /// This tile read where a walk proved it inside its run: a window masked only past the run's
    /// end ([`Overhang::Run`]) reads unchecked, any other tile as it is. Its scales are read as
    /// they are.
    #[allow(dead_code)] // Reached through its expand, from `Payload::inside_runs`.
    pub(crate) fn inside_run(&self) -> Tile {
        cubecl::unexpanded!()
    }
}

impl TileExpand {
    pub(crate) fn __expand_inside_run_method(&self, _scope: &Scope) -> TileExpand {
        let mut tile = self.clone();
        if let TileKindExpand::Memory(memory) = &mut tile.kind
            && memory.access.overhang == Overhang::Run
        {
            memory.access.overhang = Overhang::Fits;
        }
        tile
    }
}

#[cube]
impl Memory {
    /// The bound a walk reads off dim `p`: the window's, or its run's length where an index placed
    /// a run along it, so a walk over the window steps the run alone.
    pub(crate) fn walked_bound(&self, #[comptime] p: usize) -> u32 {
        let bound = self.window.bound.at(p);
        let runs = self.runs_along(p);
        if comptime!(runs) {
            bound.minus(self.window.origin.at(p).cast::<u32>())
        } else {
            bound
        }
    }
}

impl Memory {
    /// Whether an index placed a run along dim `p` of this window.
    pub(crate) fn runs_along(&self, _p: usize) -> comptime_type!(bool) {
        cubecl::unexpanded!()
    }
}

impl MemoryExpand {
    pub(crate) fn __expand_runs_along_method(&self, _scope: &Scope, p: usize) -> bool {
        match &self.index {
            ComptimeOptionExpand::Some(table) => {
                table.placed
                    && matches!(table.indexing, Indexing::Run(axis)
                        if placed_dim(&self.projection, axis) == Some(p))
            }
            ComptimeOptionExpand::None => false,
        }
    }
}

impl Tile {
    /// This operand as a slot plans its stage: who moves it, the axes it spans, and the keys of a
    /// table it has yet to read.
    pub(crate) fn stage_operand(&self) -> comptime_type!(StageOperand) {
        cubecl::unexpanded!()
    }
}

impl TileExpand {
    pub(crate) fn __expand_stage_operand_method(&self, scope: &Scope) -> StageOperand {
        StageOperand {
            delivery: self.clone().__expand_delivery_method(scope),
            space: self.place.space.clone(),
            keys: self.pending_keys(),
        }
    }

    /// The keys of a table this tile has yet to read; none where it has none, or has read it.
    pub(crate) fn pending_keys(&self) -> Vec<Axis> {
        match &self.kind {
            TileKindExpand::Memory(MemoryExpand {
                index: ComptimeOptionExpand::Some(table),
                ..
            }) if !table.placed => table.keys.clone(),
            _ => Vec::new(),
        }
    }
}

impl Tile {
    /// This tile shaped as one region of `level` holds it, for a stage: one wide along the axis
    /// its table places where `level` cuts the table's keys, and so are the scales it carries
    /// that span it. Its window is not moved; the fill of each region places it.
    pub(crate) fn shaped_for(&self, _level: Level) -> Tile {
        cubecl::unexpanded!()
    }
}

impl TileExpand {
    pub(crate) fn __expand_shaped_for_method(&self, _scope: &Scope, level: Level) -> TileExpand {
        let TileKindExpand::Memory(MemoryExpand {
            index: ComptimeOptionExpand::Some(table),
            ..
        }) = &self.kind
        else {
            return self.clone();
        };
        let Indexing::Coordinate(axis) = table.indexing else {
            return self.clone();
        };
        if table.placed || !table.keys.iter().any(|key| level.count(*key).is_some()) {
            return self.clone();
        }
        self.one_wide(axis)
    }

    /// This tile's place, and its spanning scales', one wide along `axis`; its window unmoved.
    fn one_wide(&self, axis: Axis) -> TileExpand {
        let mut shaped = self.clone();
        if shaped.place.space.contains(axis) {
            shaped.place.space = Level::every(&[(axis, 1)]).child(&shaped.place.space);
        }
        shaped.factor = self.factor.spanning(axis, |scale| scale.one_wide(axis));
        shaped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);
    const EXPERT: Axis = Axis(3);

    fn rows(tile: usize) -> Level {
        Level::every(&[(M, tile)])
    }

    #[test]
    fn a_table_is_read_at_the_first_level_cutting_its_keys() {
        let levels = [Level::every(&[(N, 4)]), rows(2), rows(1)];
        let read = Indexing::Coordinate(EXPERT).read_at(&[M], &levels);
        assert_eq!(read, Ok(Some(1)));
    }

    /// No level cuts the keys, so they are one tile: the table's first entry is the one, read as
    /// the tile is served, as a grouped product of a single group reads its one run.
    #[test]
    fn a_table_no_level_cuts_is_read_as_the_tile_is_served() {
        let levels = [Level::every(&[(N, 4)])];
        let read = Indexing::Coordinate(EXPERT).read_at(&[M], &levels);
        assert_eq!(read, Ok(None));
        let walked = [Level::every(&[(K, 4)])];
        assert_eq!(Indexing::Run(K).read_at(&[M], &walked), Ok(None));
    }

    #[test]
    fn keys_cut_at_two_levels_are_refused() {
        let levels = [rows(1), Level::every(&[(N, 1)])];
        let read = Indexing::Coordinate(EXPERT).read_at(&[M, N], &levels);
        assert_eq!(read, Err(Refusal::IndexKeysCutApart(vec![M, N])));
    }

    #[test]
    fn a_coordinate_a_level_cuts_is_refused() {
        let levels = [rows(1), Level::every(&[(EXPERT, 1)])];
        let read = Indexing::Coordinate(EXPERT).read_at(&[M], &levels);
        assert_eq!(read, Err(Refusal::IndexedAxisCut(EXPERT)));
    }

    /// A run is placed whole: a level cutting its axis at or above the key's would have cut a
    /// window the run has not placed yet. Below it, the run is what that level walks.
    #[test]
    fn a_run_cut_before_it_is_placed_is_refused() {
        let group = Level::every(&[(EXPERT, 1)]);
        let above = [rows(2), group.clone()];
        let below = [group, rows(2)];
        let run = Indexing::Run(M);
        assert_eq!(
            run.read_at(&[EXPERT], &above),
            Err(Refusal::IndexedAxisCut(M))
        );
        assert_eq!(run.read_at(&[EXPERT], &below), Ok(Some(0)));
    }

    #[test]
    fn a_coordinate_keyed_by_itself_is_refused() {
        let spec = TileSpec::direct(&[EXPERT, K, N]);
        let refused = Indexing::Coordinate(EXPERT).validate(&spec, 1, &[EXPERT]);
        assert_eq!(refused, Err(Refusal::IndexKeyedByItsAxis(EXPERT)));
        let unkeyed = Indexing::Coordinate(EXPERT).validate(&spec, 1, &[]);
        assert_eq!(unkeyed, Err(Refusal::IndexKeyedByItsAxis(EXPERT)));
    }

    #[test]
    fn a_paged_operand_is_refused() {
        let mut spec = TileSpec::direct(&[EXPERT, K, N]);
        spec.paging = Some(Paging {
            sequence: EXPERT,
            position: K,
            page_size: 4,
        });
        let refused = Indexing::Coordinate(EXPERT).validate(&spec, 1, &[M]);
        assert_eq!(refused, Err(Refusal::IndexOfAStatedLayout));
    }

    /// The innermost dim may be placed where the operand reads it one value a line: a run along
    /// the rows of a transposed operand. Read in wider lines, or packed, a line would straddle two
    /// entries.
    #[test]
    fn the_innermost_dim_is_placed_only_one_value_a_line() {
        let spec = TileSpec::direct(&[N, K]);
        let run = Indexing::Run(K);
        assert_eq!(run.validate(&spec, 1, &[M]), Ok(()));
        assert_eq!(
            run.validate(&spec, 4, &[M]),
            Err(Refusal::IndexAlongALine(K))
        );
        let packed =
            TileSpec::direct(&[N, K]).packed(Field::from(cubecl::quant::scheme::QuantValue::Q8S));
        assert_eq!(
            run.validate(&packed, 1, &[M]),
            Err(Refusal::IndexAlongALine(K))
        );
    }

    /// A packed weight blocks its columns over two axes, so its projection is not direct; the
    /// expert is still one dim of its own, which is all a coordinate moves.
    #[test]
    fn a_coordinate_places_a_whole_dim_of_a_gathered_operand_and_a_run_does_not() {
        let block = Axis(4);
        let spec = TileSpec::new(Projection::new(
            &[EXPERT, K, N, block],
            &[
                PhysicalAxisMap::of(EXPERT),
                PhysicalAxisMap::of(K),
                PhysicalAxisMap::disjoint(&[(N, 4), (block, 1)]),
            ],
        ));
        assert_eq!(
            Indexing::Coordinate(EXPERT).validate(&spec, 1, &[M]),
            Ok(())
        );
        assert_eq!(
            Indexing::Run(EXPERT).validate(&spec, 1, &[M]),
            Err(Refusal::IndexOfAStatedLayout)
        );
    }
}

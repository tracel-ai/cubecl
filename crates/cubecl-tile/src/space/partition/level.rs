//! One [`Level`]: which dimensions a loop steps, in tiles of what size, how many, and who takes
//! them.

use super::{
    ComputeScope, Count, Coverage, CubeAxis, CubeOrder, GridCount, PlaneOrder, Spread, UnitCube,
};
use crate::{Dimension, DimensionMap, Extent, Space};

/// One decomposition level of a space: who takes its tiles and how it cuts each dimension it names.
/// Unnamed dimensions are handed down whole.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Level {
    coverage: Coverage,
    cuts: DimensionMap<Cut>,
    /// Dimensions a cube level hands one tile of, riding the grid's `Z`
    /// ([`batching`](Level::batching)).
    batches: Vec<Dimension>,
    shared_by: Option<usize>,
    fillers: usize,
    order: CubeOrder,
    /// Who across the whole launch takes this level's tiles, where it spans the launch rather than
    /// one cube.
    launch_span: Option<LaunchSpan>,
    /// Dimensions whose tiles go out last first ([`Levels::reversed`](crate::Levels::reversed)).
    reversed: Vec<Dimension>,
}

/// Who takes the tiles of a level spanning the launch.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LaunchSpan {
    /// The launch's planes, in this order ([`spanning_launch`](Level::spanning_launch)).
    Planes(PlaneOrder),
    /// The launch's units, grouped in these cubes
    /// ([`units_spanning_launch`](Level::units_spanning_launch)).
    Units(UnitCube),
}

/// One dimension of a level: the tile it steps in, how many, and how its instances spread them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Cut {
    pub(crate) tile: usize,
    pub(crate) count: Count,
    pub(crate) spread: Spread,
}

impl Level {
    /// The walk level over a region in tiles of `tile`, taking every one of them.
    pub fn every(tile: &[(Dimension, usize)]) -> Level {
        let cuts: Vec<_> = tile
            .iter()
            .map(|&(dimension, tile)| (dimension, tile, Count::All, Spread::Contiguous))
            .collect();
        Level::new(Coverage::Walk, &cuts)
    }

    /// A level covered as `coverage` says over `cuts`, each `(dimension, tile, count, spread)`.
    pub(crate) fn new(coverage: Coverage, cuts: &[(Dimension, usize, Count, Spread)]) -> Level {
        for (i, &(dimension, ..)) in cuts.iter().enumerate() {
            assert!(
                !cuts[..i].iter().any(|&(a, ..)| a == dimension),
                "Level: {dimension:?} is named twice; a level states each of its dimensions once"
            );
        }
        for &(dimension, tile, count, _) in cuts {
            assert!(tile > 0, "Level: {dimension:?} has a tile of nothing");
            match (coverage, count) {
                (
                    Coverage::Distribute(ComputeScope::Cube)
                    | Coverage::Distribute(ComputeScope::Plane),
                    Count::AllAcross(_),
                ) => {}
                (_, Count::AllAcross(_)) => {
                    panic!(
                        "Level: {dimension:?} is distributed across workers in runs, which only cubes \
                         and planes take"
                    )
                }
                (
                    Coverage::Distribute(ComputeScope::Unit)
                    | Coverage::Distribute(ComputeScope::Plane)
                    | Coverage::Distribute(ComputeScope::PlaneGroup { .. }),
                    Count::All,
                ) => panic!(
                    "Level: {dimension:?} takes every tile on a plane's workers, whose count is the \
                     device's; state how many"
                ),
                (Coverage::Distribute(ComputeScope::Unit), Count::InTurns { .. }) => assert!(
                    matches!(
                        cuts.iter()
                            .find(|&&(a, ..)| a == dimension)
                            .map(|&(.., s)| s),
                        Some(Spread::Interleaved)
                    ),
                    "Level: {dimension:?} is taken in turns by units, so they take it interleaved"
                ),
                (_, Count::InTurns { .. }) => panic!(
                    "Level: {dimension:?} is taken in turns by units, which only a units level hands \
                     them"
                ),
                _ => {}
            }
        }
        let distributed = cuts
            .iter()
            .any(|&(_, _, count, _)| matches!(count, Count::InTurns { .. }));
        assert!(
            !distributed
                || (coverage == Coverage::Distribute(ComputeScope::Unit) && cuts.len() == 1),
            "Level: a dimension taken in turns by units takes all of them, so the level distributes no \
             other dimension to the units; it distributes {}",
            cuts.len()
        );
        let cuts: Vec<_> = cuts
            .iter()
            .map(|&(dimension, tile, count, spread)| {
                (
                    dimension,
                    Cut {
                        tile,
                        count,
                        spread,
                    },
                )
            })
            .collect();
        Level {
            coverage,
            cuts: DimensionMap::new(&cuts),
            batches: Vec::new(),
            shared_by: None,
            fillers: 0,
            order: CubeOrder::RowMajor,
            launch_span: None,
            reversed: Vec::new(),
        }
    }

    /// The one dimension of a copy's lines, which its level deals
    /// ([`copied_by`](Level::copied_by)).
    pub(crate) const COPIED: Dimension = Dimension(0);

    /// The level a copy into or out of a window held by `holder` deals its lines by: line
    /// `w + t·n` to unit `w` of the `n` units of one instance of `holder`, `units` of them where
    /// the launch states it, so neighbouring units take neighbouring lines at once; every line to
    /// the one unit of a window a unit holds.
    pub(crate) fn copied_by(holder: ComputeScope, units: Option<usize>) -> Level {
        match holder {
            ComputeScope::Unit => Level::every(&[(Level::COPIED, 1)]),
            within => Level::new(
                Coverage::Distribute(ComputeScope::Unit),
                &[(
                    Level::COPIED,
                    1,
                    Count::InTurns {
                        tiles: None,
                        within,
                        units,
                    },
                    Spread::Interleaved,
                )],
            ),
        }
    }

    /// The level a copy into or out of a window the cube holds deals its lines by
    /// ([`copied_by`](Level::copied_by)), `units` of them where a binding states its launch's count
    /// ([`TileSpec::units`](crate::TileSpec), `0` where it states none).
    pub(crate) fn copied_by_cube(units: usize) -> Level {
        Level::copied_by(ComputeScope::Cube, (units > 0).then_some(units))
    }

    /// The level a copy into a cube's stage deals its lines by where the cube sets `planes` planes
    /// aside to fill its stages: to their units alone, the cube's last.
    pub(crate) fn copied_by_fillers(planes: usize) -> Level {
        let mut level = Level::copied_by(ComputeScope::Cube, None);
        level.fillers = planes;
        level
    }

    /// This copy's level narrowed to `holder`'s units where they are fewer: a plane's window of a
    /// buffer the cube copies is the plane's to copy.
    pub(crate) fn held_by(self, holder: ComputeScope) -> Level {
        match holder < self.copied_within() {
            true => Level::copied_by(holder, None),
            false => self,
        }
    }

    /// The scope whose units a copy's level deals its lines to: a unit where it walks them all.
    pub(crate) fn copied_within(&self) -> ComputeScope {
        match self.count(Level::COPIED) {
            Some(Count::InTurns { within, .. }) => within,
            _ => ComputeScope::Unit,
        }
    }

    /// The units a copy's level deals its lines to: the scope they are the units of, how many where
    /// the launch states it, and the filling planes at the end of the cube they are confined to.
    pub(crate) fn copied_units(&self) -> (ComputeScope, Option<usize>, usize) {
        match self.count(Level::COPIED) {
            Some(Count::InTurns { within, units, .. }) => (within, units, self.fillers),
            _ => (ComputeScope::Unit, Some(1), 0),
        }
    }

    /// The turns each unit takes over `lines` of a copy dealt by this level, the last of them
    /// ragged, where the count of its units is known.
    pub(crate) fn copy_turns(&self, lines: usize) -> Option<usize> {
        match self.count(Level::COPIED) {
            Some(Count::InTurns { units, .. }) => units.map(|units| lines.div_ceil(units)),
            _ => Some(lines),
        }
    }

    /// The plane level over every tile of `tile`'s dimensions, each `(dimension, tile)`, dealt to
    /// the planes of the whole launch in `order` rather than to the planes of one cube: the
    /// outermost level of a partitioning with no cube level, whose cubes are only how the launch
    /// groups its planes.
    pub(crate) fn spanning_launch(tile: &[(Dimension, usize)], order: PlaneOrder) -> Level {
        assert!(
            !tile.is_empty(),
            "Level::spanning_launch: the launch's planes take the tiles of at least one dimension"
        );
        for dimension in order.block_dimensions() {
            assert!(
                tile.iter().any(|&(a, _)| a == dimension),
                "Level::spanning_launch: the order's block names {dimension:?}, which the level does not"
            );
        }
        assert!(
            tile.len() > 1 || !order.block_order().swizzles(),
            "Level::spanning_launch: the blocks of one dimension have one order; a swizzle takes two"
        );
        // A capped launch's planes take the tiles in turns: a whole launch of planes apart.
        let spread = match order.cap() {
            Some(_) => Spread::Interleaved,
            None => Spread::Contiguous,
        };
        let cuts: Vec<_> = tile
            .iter()
            .map(|&(dimension, tile)| {
                assert!(tile > 0, "Level: {dimension:?} has a tile of nothing");
                (
                    dimension,
                    Cut {
                        tile,
                        count: Count::All,
                        spread,
                    },
                )
            })
            .collect();
        Level {
            coverage: Coverage::Distribute(ComputeScope::Plane),
            cuts: DimensionMap::new(&cuts),
            batches: Vec::new(),
            shared_by: None,
            fillers: 0,
            order: CubeOrder::RowMajor,
            launch_span: Some(LaunchSpan::Planes(order)),
            reversed: Vec::new(),
        }
    }

    /// The unit level over every tile of `tile`'s dimensions, each `(dimension, tile)`, one tile to
    /// each unit of the whole launch, the units grouped in cubes of `cube`: the outermost level of
    /// a partitioning with no cube or plane level.
    pub(crate) fn units_spanning_launch(tile: &[(Dimension, usize)], cube: UnitCube) -> Level {
        assert!(
            cube.units() > 0,
            "Level::units_spanning_launch: a cube holds at least one unit"
        );
        let order = cube.order();
        Level {
            coverage: Coverage::Distribute(ComputeScope::Unit),
            launch_span: Some(LaunchSpan::Units(cube)),
            ..Level::spanning_launch(tile, order)
        }
    }

    /// The order the launch's planes take this level's tiles in, where it is a plane level
    /// spanning the launch.
    pub fn plane_order(&self) -> Option<&PlaneOrder> {
        match &self.launch_span {
            Some(LaunchSpan::Planes(order)) => Some(order),
            _ => None,
        }
    }

    /// The cubes the launch's units are grouped in, where it is a unit level spanning the launch.
    pub(crate) fn unit_cube(&self) -> Option<UnitCube> {
        match self.launch_span {
            Some(LaunchSpan::Units(cube)) => Some(cube),
            _ => None,
        }
    }

    /// The order the launch's planes or units take this level's tiles in, where it spans the
    /// launch: one tile a unit, in row order, for units.
    pub(crate) fn launch_order(&self) -> Option<PlaneOrder> {
        match &self.launch_span {
            Some(LaunchSpan::Planes(order)) => Some(order.clone()),
            Some(LaunchSpan::Units(cube)) => Some(cube.order()),
            None => None,
        }
    }

    /// Whether this level hands its tiles to the planes or units of the whole launch.
    pub fn spans_launch(&self) -> bool {
        self.launch_span.is_some()
    }

    /// One tile of each of `dimensions` per cube, on the grid's `Z` dimension.
    pub(crate) fn batching(mut self, dimensions: &[Dimension]) -> Level {
        assert!(
            self.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Level::batching: batches ride cubes; this level's tiles are taken by {:?}",
            self.coverage
        );
        for &dimension in dimensions {
            self.push(
                dimension,
                Cut {
                    tile: 1,
                    count: Count::All,
                    spread: Spread::Contiguous,
                },
            );
            self.batches.push(dimension);
        }
        self
    }

    /// Every cut's tiles read as one index, of which each of `n` workers takes a run.
    pub(crate) fn sharing(mut self, n: usize) -> Level {
        match self.coverage {
            Coverage::Distribute(ComputeScope::Cube)
            | Coverage::Distribute(ComputeScope::Plane) => {}
            Coverage::Distribute(ComputeScope::PlaneGroup { .. }) => panic!(
                "Level::sharing: a plane group's partials of one box are not merged yet; deal \
                 each group a box of its own"
            ),
            Coverage::Distribute(ComputeScope::Unit) => panic!(
                "Level::sharing: the plane's units combine in registers, which needs them in \
                 lockstep, and units holding different shares never are"
            ),
            Coverage::Walk => panic!("Level::sharing: a walk has no workers to share its tiles"),
        }
        assert!(
            self.shared_by.is_none(),
            "Level::sharing: this level already cuts its tiles as one"
        );
        for dimension in self.dimensions() {
            let cut = self.cuts.get(dimension);
            assert!(
                cut.count == Count::All && cut.spread == Spread::Contiguous,
                "Level::sharing: {dimension:?} states a count or a spread of its own, which a share \
                 of the whole has no use for"
            );
        }
        self.shared_by = Some(n);
        self
    }

    /// The stages of this walk are filled by `n` extra planes at the end of the cube.
    pub(crate) fn filling(mut self, n: usize) -> Level {
        assert!(
            self.coverage == Coverage::Walk,
            "Level::filling: only a walk's regions are staged, and this level's tiles are taken \
             by {:?}",
            self.coverage
        );
        self.fillers = n;
        self
    }

    /// The order this cube level distributes its boxes to the grid.
    pub(crate) fn distributed_in(mut self, order: CubeOrder) -> Level {
        assert!(
            self.coverage == Coverage::Distribute(ComputeScope::Cube),
            "Level::distributed_in: only a cube level is distributed to a grid, and this level's tiles are \
             taken by {:?}",
            self.coverage
        );
        self.order = order.canonicalize();
        self
    }

    /// This level handing out the tiles of each of `dimensions` from the last to the first, on top
    /// of whatever order it deals them in: the first workers launched take the last tiles.
    pub(crate) fn reversed(mut self, dimensions: &[Dimension]) -> Level {
        assert!(
            self.coverage == Coverage::Distribute(ComputeScope::Cube) || self.spans_launch(),
            "Level::reversed: only a cube level or a level spanning the launch hands its tiles out \
             in a launch order, and this level's tiles are taken by {:?}",
            self.coverage
        );
        assert!(
            self.shared_by.is_none(),
            "Level::reversed: this level cuts its tiles as one index, which has no dimension to reverse"
        );
        for &dimension in dimensions {
            assert!(
                self.cuts.contains(dimension),
                "Level::reversed: {dimension:?} is not a dimension of this level"
            );
            assert!(
                !self.reversed.contains(&dimension),
                "Level::reversed: {dimension:?} is reversed twice"
            );
            self.reversed.push(dimension);
        }
        self
    }

    fn push(&mut self, dimension: Dimension, cut: Cut) {
        assert!(
            !self.cuts.contains(dimension),
            "Level: {dimension:?} is named twice; a level states each of its dimensions once"
        );
        let mut cuts: Vec<_> = self
            .dimensions()
            .into_iter()
            .map(|a| (a, self.cuts.get(a)))
            .collect();
        cuts.push((dimension, cut));
        self.cuts = DimensionMap::new(&cuts);
    }

    /// Who takes this level's tiles.
    pub fn coverage(&self) -> Coverage {
        self.coverage
    }

    /// The dimensions this level names, in the order it named them.
    pub fn dimensions(&self) -> Vec<Dimension> {
        (0..self.cuts.len())
            .map(|i| self.cuts.dimension_at(i))
            .collect()
    }

    /// The tile this level steps `dimension` in, where it names it.
    pub fn tile(&self, dimension: Dimension) -> Option<usize> {
        self.cuts
            .contains(dimension)
            .then(|| self.cuts.get(dimension).tile)
    }

    /// How many tiles this level takes along `dimension`, where it names it.
    pub(crate) fn count(&self, dimension: Dimension) -> Option<Count> {
        self.cuts
            .contains(dimension)
            .then(|| self.cuts.get(dimension).count)
    }

    /// How the instances spread `dimension`'s tiles, where the level names it.
    pub(crate) fn spread(&self, dimension: Dimension) -> Option<Spread> {
        self.cuts
            .contains(dimension)
            .then(|| self.cuts.get(dimension).spread)
    }

    pub(crate) fn cut(&self, dimension: Dimension) -> Cut {
        self.cuts.get(dimension)
    }

    /// Whether this level distributes `dimension`'s tiles over its scope one by one.
    pub(crate) fn distributes(&self, dimension: Dimension) -> bool {
        self.coverage != Coverage::Walk && self.shared_by.is_none() && self.cuts.contains(dimension)
    }

    /// The grid axis `dimension` rides on a cube level: batches on `Z`, the others `X`, `Y`, `Z`
    /// in the order named.
    pub(crate) fn cube_axis(&self, dimension: Dimension) -> Option<CubeAxis> {
        if self.coverage != Coverage::Distribute(ComputeScope::Cube)
            || !self.cuts.contains(dimension)
        {
            return None;
        }
        if self.batches.contains(&dimension) {
            return Some(CubeAxis::Z);
        }
        let cube_axes = [CubeAxis::X, CubeAxis::Y, CubeAxis::Z];
        let position = self
            .dimensions()
            .into_iter()
            .filter(|a| !self.batches.contains(a))
            .position(|a| a == dimension)
            .expect("the dimension is named and not a batch");
        Some(cube_axes[position])
    }

    /// The workers taking this level's grid as one flat index.
    pub fn shared_by(&self) -> Option<usize> {
        self.shared_by
    }

    /// Planes that fill this level's stages.
    pub fn fillers(&self) -> usize {
        self.fillers
    }

    /// The order this level distributes its boxes to the grid.
    pub fn order(&self) -> CubeOrder {
        self.order
    }

    /// Whether this level hands out `dimension`'s tiles from the last to the first.
    pub fn is_reversed(&self, dimension: Dimension) -> bool {
        self.reversed.contains(&dimension)
    }

    /// The extent one region of this level covers along `dimension` of `space`.
    pub(crate) fn extent_in(&self, space: &Space, dimension: Dimension) -> Extent {
        match self.tile(dimension) {
            Some(tile) => Extent::Static(tile),
            None => space.extent_raw(dimension),
        }
    }

    /// The space one region of this level covers.
    pub fn child(&self, space: &Space) -> Space {
        Space::from_extents(
            &space
                .dimensions()
                .map(|dimension| (dimension, self.extent_in(space, dimension)))
                .collect::<Vec<_>>(),
        )
    }

    /// Whether this level's tile on `dimension` fails to divide the extent `space` hands it.
    /// Static extents only.
    pub(crate) fn overhangs(&self, space: &Space, dimension: Dimension) -> bool {
        let Some(tile) = self.tile(dimension) else {
            return false;
        };
        let cut = !space.extent(dimension).is_multiple_of(tile);
        match self.launch_order() {
            None => cut,
            Some(order) => cut || order.overruns(self, space, dimension),
        }
    }

    /// Tiles along `dimension` of `space` at this level, a partial tile counted; static extents
    /// only.
    pub(crate) fn tiles(&self, space: &Space, dimension: Dimension) -> usize {
        match self.count(dimension) {
            None => 1,
            Some(count) => count.tiles(space.extent(dimension), self.cut(dimension).tile),
        }
    }

    /// Tiles along `dimension` where the count is comptime.
    pub(crate) fn tiles_const(&self, space: &Space, dimension: Dimension) -> Option<usize> {
        match self.count(dimension) {
            None => Some(1),
            Some(count) => count.tiles_const(space.extent_raw(dimension), self.cut(dimension).tile),
        }
    }

    /// What a walk counts along `dimension`.
    pub(crate) fn grid(&self, dimension: Dimension) -> GridCount {
        match self.count(dimension) {
            None => GridCount::Const(1),
            Some(count) => count.grid(self.cut(dimension).tile),
        }
    }

    /// How many workers `dimension` is distributed out to at this level, where comptime.
    pub(crate) fn instances_along(&self, space: &Space, dimension: Dimension) -> Option<usize> {
        match self.count(dimension) {
            None => Some(1),
            Some(Count::InTurns { units, .. }) => units,
            Some(count) => match count.stated() {
                Some(n) => Some(n),
                None if !space.contains(dimension) => None,
                None => self.tiles_const(space, dimension),
            },
        }
    }
}

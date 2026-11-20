//! [`VectorTile`]: the values one vector load brings, as a tile.
//!
//! An operand says how it is stored, and the kernel picks how wide it loads; the tile a load covers
//! follows from both. A plain buffer is loaded in a run along its innermost axis. A buffer stored
//! in tiles is loaded a stored tile at a time: its finest tiles, fused per dimension, until they
//! hold the load's values, so an NVFP4 buffer whose word runs along `K` and whose next pieces are
//! two words along `K` and two columns loads 32 values as 16 along `K` by 2 along `N`.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::{
    Coords, Dimension, Extent, Space, TileMisfit,
    algebra::{Integer, IntegerExpand},
    comptime_only,
};

/// How much of a buffer one unit reads whole, the next unit reading on from where it stops: one
/// load, one memory transaction, or one stored tile. Which is fastest depends on the device, so a
/// read split across units states it; how many loads it holds follows from how the buffer is
/// stored ([`VectorTile::run_loads`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Run {
    Load,
    Transaction,
    StoredTile,
}

/// The values one vector load brings, as a tile: its extents, finest first, one per dimension.
///
/// ```ignore
/// // a plain row-major buffer, four wide
/// VectorTile::new(&[], N, 4)                                  // [(N, 4)]
/// // NVFP4 values: a word of 8 along K, 2 words along K, 2 columns; 32 values a load
/// VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32)           // [(K, 16), (N, 2)]
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorTile {
    extents: Vec<(Dimension, usize)>,
    /// The stored tiles above one load, finest first: the pieces a buffer lays its loads in.
    above: Vec<(Dimension, usize)>,
}

impl VectorTile {
    /// The tile a load of `values` values covers over a buffer stored in `stored` tiles, finest
    /// first ([`TileSpec::stored_tiles`](crate::TileSpec::stored_tiles)): its finest tiles, fused
    /// where one dimension continues, until they hold `values`. Nothing stated, a run of `values`
    /// along `innermost`.
    ///
    /// # Errors
    ///
    /// A load that would cut a stored tile, one wider than every stored tile together, or one
    /// whose dimension comes back after another, which is no longer one extent per dimension.
    pub fn new(
        stored: &[(Dimension, usize)],
        innermost: Dimension,
        values: usize,
    ) -> Result<Self, TileMisfit> {
        let stored: Vec<(Dimension, usize)> = stored
            .iter()
            .copied()
            .filter(|&(_, count)| count > 1)
            .collect();
        if stored.is_empty() || values == 1 {
            return Ok(Self {
                extents: vec![(innermost, values)],
                above: stored,
            });
        }
        let mut extents: Vec<(Dimension, usize)> = Vec::new();
        let mut held = 1;
        let mut fused = 0;
        for &(dimension, count) in &stored {
            if held == values {
                break;
            }
            fused += 1;
            if !(values / held).is_multiple_of(count) {
                return Err(TileMisfit::Overshoots {
                    dimension,
                    wanted: values / held,
                    piece: count,
                });
            }
            held *= count;
            match extents.last().copied() {
                Some((last, extent)) if last == dimension => {
                    *extents.last_mut().expect("just read") = (dimension, extent * count);
                }
                Some((last, _)) if extents.iter().any(|&(a, _)| a == dimension) => {
                    return Err(TileMisfit::Interleaved {
                        wanted: dimension,
                        found: last,
                    });
                }
                _ => extents.push((dimension, count)),
            }
        }
        if held != values {
            return Err(TileMisfit::RunsOut {
                wanted: extents
                    .last()
                    .map_or(innermost, |&(dimension, _)| dimension),
            });
        }
        Ok(Self {
            extents,
            above: stored[fused..].to_vec(),
        })
    }

    /// A run of `values` along `dimension`: how a plain buffer is loaded.
    pub fn run(dimension: Dimension, values: usize) -> Self {
        Self {
            extents: vec![(dimension, values)],
            above: Vec::new(),
        }
    }

    /// The loads of a window spanning `space` in the order its buffer lays them, as the digits of
    /// a load's number, outermost first: each `(position, count, step)` moves the load `step`
    /// loads along the dimension at `position`, `count` times. The stored tiles above one load come
    /// first, finest first, each as far as the window holds it whole; then each dimension's rest, the
    /// last dimension fastest. A buffer stored in no tiles is counted with the last dimension fastest.
    pub(crate) fn order(&self, space: &Space) -> Vec<(usize, usize, usize)> {
        let counts = self.counts(space);
        let mut taken = vec![1; space.rank()];
        let mut whole = vec![true; space.rank()];
        let mut finest_first = Vec::new();
        for &(dimension, count) in self
            .above
            .iter()
            .filter(|&&(dimension, _)| space.contains(dimension))
        {
            let p = space.position(dimension);
            whole[p] = whole[p] && counts[p].is_multiple_of(taken[p] * count);
            if whole[p] {
                finest_first.push((p, count, taken[p]));
                taken[p] *= count;
            }
        }
        for p in (0..space.rank()).rev() {
            finest_first.push((p, counts[p] / taken[p], taken[p]));
        }
        finest_first.reverse();
        finest_first
    }

    /// The extents, finest first.
    pub fn extents(&self) -> &[(Dimension, usize)] {
        &self.extents
    }

    /// Values one load holds along its finest dimension: one line of it, the whole load where it
    /// spans one dimension. A reader walking lines along that dimension takes `values() / run()` of
    /// them a load.
    pub(crate) fn run_length(&self) -> usize {
        self.extents.first().map_or(1, |&(_, run)| run)
    }

    /// The box one load covers, a dimension to each extent.
    pub fn space(&self) -> Space {
        Space::new(&self.extents)
    }

    /// How many values the load brings.
    pub fn values(&self) -> usize {
        self.extents.iter().map(|&(_, extent)| extent).product()
    }

    /// The dimension a one-dimension load runs along and its width, for a `reader` that places a
    /// load along one dimension.
    ///
    /// # Panics
    ///
    /// When the load spans several dimensions, naming `reader`.
    pub(crate) fn along_one_dimension(&self, reader: &str) -> (Dimension, usize) {
        match self.extents.as_slice() {
            &[one] => one,
            extents => panic!(
                "{reader} places a load along one dimension, and this operand loads {extents:?}: \
                 read it with a narrower vector, or through a path that places a tile"
            ),
        }
    }

    /// How far one load reaches along `dimension`: its extent there, `1` off the dimensions it
    /// spans.
    pub fn extent_along(&self, dimension: Dimension) -> usize {
        self.extents
            .iter()
            .find(|&&(a, _)| a == dimension)
            .map_or(1, |&(_, extent)| extent)
    }

    /// How many values of one load lie between two neighbours along `dimension`: the product of the
    /// extents finer than `dimension`'s. `0` off the dimensions the load spans.
    pub(crate) fn step_along(&self, dimension: Dimension) -> usize {
        let mut step = 1;
        for &(a, extent) in &self.extents {
            if a == dimension {
                return step;
            }
            step *= extent;
        }
        0
    }

    /// The loads along each dimension of `space` a `run` holds, multiplied up from one load: the stored
    /// tiles above it, finest first, each as far as the window holds it whole, until the run is
    /// complete; past them, loads along the innermost dimension, until a transaction of
    /// `transaction_bytes` over `bits`-bit values is full. `None` for a stored tile where the
    /// buffer stores none above one load.
    pub fn run_loads(
        &self,
        run: Run,
        space: &Space,
        bits: usize,
        transaction_bytes: usize,
    ) -> Option<Vec<(Dimension, usize)>> {
        let counts = self.counts(space);
        let mut taken = vec![1; space.rank()];
        let load_bytes = self.values() * bits / 8;
        let full =
            |taken: &[usize]| load_bytes * taken.iter().product::<usize>() >= transaction_bytes;
        if run != Run::Load {
            let mut whole = vec![true; space.rank()];
            for &(dimension, count) in self
                .above
                .iter()
                .filter(|&&(dimension, _)| space.contains(dimension))
            {
                if run == Run::Transaction && full(&taken) {
                    break;
                }
                let p = space.position(dimension);
                // A transaction takes a piece only as far as it fills: the fewest of its parts
                // that do, a factor of its count. A stored tile takes it whole.
                let held = load_bytes * taken.iter().product::<usize>();
                let part = match run {
                    Run::Transaction => (1..=count)
                        .filter(|&part| count.is_multiple_of(part))
                        .find(|&part| held * part >= transaction_bytes)
                        .unwrap_or(count),
                    Run::Load | Run::StoredTile => count,
                };
                whole[p] = whole[p] && counts[p].is_multiple_of(taken[p] * part);
                if whole[p] {
                    taken[p] *= part;
                }
            }
        }
        match run {
            Run::StoredTile if taken.iter().all(|&loads| loads == 1) => return None,
            Run::Transaction => {
                let innermost = space.rank() - 1;
                while !full(&taken) && counts[innermost].is_multiple_of(taken[innermost] * 2) {
                    taken[innermost] *= 2;
                }
            }
            Run::Load | Run::StoredTile => {}
        }
        Some(space.dimensions().zip(taken).collect())
    }

    /// How far along `dimension` the `line`-th of these loads of a window spanning `space` sits from
    /// the window's first value, the loads counted in the order the buffer lays them
    /// ([`order`](VectorTile::order)): the place [`start`](VectorTile::start) gives, known at
    /// expansion.
    pub(crate) fn offset_in(&self, line: usize, space: &Space, dimension: Dimension) -> usize {
        let p = space.position(dimension);
        let mut rest = line;
        let mut loads = 0;
        for (at, count, step) in self.order(space).into_iter().rev() {
            if at == p {
                loads += (rest % count) * step;
            }
            rest /= count;
        }
        loads * self.extent_along(dimension)
    }

    /// How far along `dimension` the value at `position` of one load sits from the load's first:
    /// the position unravelled over the load's extents, finest first. `20` of a 16 by 2 load along
    /// `K` then `N` is 4 along `K` and 1 along `N`.
    pub(crate) fn offset_along(&self, position: usize, dimension: Dimension) -> usize {
        let mut rest = position;
        for &(a, extent) in &self.extents {
            if a == dimension {
                return rest % extent;
            }
            rest /= extent;
        }
        0
    }

    /// How many loads an edge of `edge` values along `dimension` holds, rounded up: an edge cuts
    /// whole loads, or is the dimension's whole `extent`, since a padded stage's innermost extent
    /// need not fill its last load.
    ///
    /// # Panics
    ///
    /// An edge that is neither, so the next region would start mid-load. A `Dynamic` dimension has
    /// no extent to be cut whole, so it owes the divisibility.
    pub(crate) fn loads_in(&self, dimension: Dimension, edge: usize, extent: Extent) -> usize {
        let w = self.extent_along(dimension);
        assert!(
            edge.is_multiple_of(w) || matches!(extent, Extent::Static(x) if x == edge),
            "Memory::at: the edge {edge} along {dimension:?} is neither a whole number of loads {w} \
             wide there nor the dimension's whole extent ({extent:?}), so a step would start mid-load"
        );
        edge.div_ceil(w)
    }

    /// How many loads fit along each dimension of `space`, rounded up: a padded stage's innermost
    /// extent need not fill whole loads, and a checked read's box must include the partial last
    /// one.
    pub(crate) fn counts(&self, space: &Space) -> Vec<usize> {
        space
            .dimensions()
            .map(|dimension| {
                space
                    .extent(dimension)
                    .div_ceil(self.extent_along(dimension))
            })
            .collect()
    }

    /// How much of each of a buffer's `rank` axes one load covers, over a buffer stored in
    /// `stored` tiles, finest first, whose trailing axes `labels` name: a stated tile the load
    /// takes whole is covered entirely, and a plain buffer's innermost axis by the load's width.
    /// `1` elsewhere.
    ///
    /// A buffer read in loads is the same buffer with each axis's extent divided by what one load
    /// covers of it and every stride counted in loads: `extent / part` and `stride * part /
    /// values`, which for a plain buffer is its innermost axis in lines and the rest's strides
    /// divided by the width.
    pub(crate) fn parts(
        &self,
        stored: &[(Dimension, usize)],
        labels: &[Dimension],
        rank: usize,
    ) -> Vec<usize> {
        let mut parts = vec![1; rank];
        if stored.is_empty() || self.values() == 1 {
            if let Some(last) = parts.last_mut() {
                *last = self.values();
            }
            return parts;
        }
        let mut taken = 1;
        for (&(_, count), axis) in stored.iter().zip(Self::stated_axes(stored, labels, rank)) {
            if count == 1 {
                continue;
            }
            if taken == self.values() {
                break;
            }
            parts[axis] = count;
            taken *= count;
        }
        parts
    }

    /// The extent of each of a buffer's `rank` axes that is a stated tile, over a buffer stored
    /// in `stored` tiles whose trailing axes `labels` name; `None` for an axis the buffer's own
    /// shape sets. A stated tile's extent is fixed where the kernel is written, so a read over it
    /// divides by a constant rather than by an axis read off the buffer.
    pub(crate) fn stated_extents(
        stored: &[(Dimension, usize)],
        labels: &[Dimension],
        rank: usize,
    ) -> Vec<Option<usize>> {
        let mut extents = vec![None; rank];
        for (&(_, count), axis) in stored.iter().zip(Self::stated_axes(stored, labels, rank)) {
            extents[axis] = Some(count);
        }
        extents
    }

    /// The axis of a buffer of `rank` axes, whose trailing axes `labels` name, each of `stored`'s
    /// tiles is: the `j`-th finest stated tile of a dimension is its `j`-th axis from the end.
    fn stated_axes(stored: &[(Dimension, usize)], labels: &[Dimension], rank: usize) -> Vec<usize> {
        let unlabelled = rank - labels.len();
        let mut from_end: Vec<(Dimension, usize)> = Vec::new();
        stored
            .iter()
            .map(|&(dimension, _)| {
                let nth = match from_end.iter_mut().find(|(a, _)| *a == dimension) {
                    Some((_, seen)) => {
                        *seen += 1;
                        *seen
                    }
                    None => {
                        from_end.push((dimension, 1));
                        1
                    }
                };
                let axis = (0..labels.len())
                    .rev()
                    .filter(|&d| labels[d] == dimension)
                    .nth(nth - 1)
                    .expect("a stated tile is an axis of the buffer");
                unlabelled + axis
            })
            .collect()
    }
}

#[cube]
impl VectorTile {
    fn count_of(#[comptime] space: &Space, #[comptime] tile: &VectorTile) -> u32 {
        comptime!(tile.counts(space).iter().product::<usize>() as u32).runtime()
    }

    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    fn start_of(
        line: u32,
        #[comptime] space: &Space,
        #[comptime] tile: &VectorTile,
    ) -> Coords<u32> {
        let rank = space.rank();
        let order = tile.order(space);
        let digits = Coords::constant(comptime!(
            order.iter().map(|&(_, count, _)| count).collect::<Vec<_>>()
        ))
        .unravel(line);
        // Each digit moved to its place along its dimension, in loads.
        let mut moves = Coords::<u32>::new();
        #[unroll]
        for d in 0..order.len() {
            let step = comptime!(order[d].2 as u32);
            moves.push(digits.at(d).times(step));
        }
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let along = comptime!(
                order
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(at, ..))| at == p)
                    .map(|(d, _)| d)
                    .collect::<Vec<_>>()
            );
            let extent = comptime!(tile.extent_along(space.dimension_at(p)) as u32);
            coords.push(moves.sum(along).times(extent));
        }
        coords
    }

    fn index_of(
        coords: &Coords<u32>,
        #[comptime] space: &Space,
        #[comptime] tile: &VectorTile,
    ) -> CoordsDyn {
        let rank = space.rank();
        let mut index = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            let extent = comptime!(tile.extent_along(space.dimension_at(p)) as u32);
            index.push(coords.at(p).divided_by(extent));
        }
        index
    }
}

/// `load.count(space)`, `load.start(line, space)`, `load.index(&coords, space)`: written out by
/// hand, since a compile-time value has no kernel-side twin for `#[cube]` to hang a method on.
impl VectorTile {
    /// How many loads a window spanning `space` holds.
    pub fn count(&self, space: &Space) -> u32 {
        VectorTile::count_of(space, self)
    }

    /// The first value of the `line`-th load of a window spanning `space`, one coordinate per
    /// dimension, the loads counted in the order the buffer lays them ([`order`](VectorTile::order)):
    /// neighbouring lines are neighbouring loads of one stored tile, then of the next.
    pub fn start(&self, line: u32, space: &Space) -> Coords<u32> {
        VectorTile::start_of(line, space, self)
    }

    /// Where the load starting at `coords` sits in a view of a window spanning `space` read in
    /// these loads: each coordinate divided by the load's extent along its dimension.
    pub fn index(&self, coords: &Coords<u32>, space: &Space) -> CoordsDyn {
        VectorTile::index_of(coords, space, self)
    }

    pub fn __expand_count_method(&self, scope: &Scope, space: &Space) -> NativeExpand<u32> {
        VectorTile::__expand_count_of(scope, space, self)
    }

    pub fn __expand_start_method(
        &self,
        scope: &Scope,
        line: NativeExpand<u32>,
        space: &Space,
    ) -> <Coords<u32> as CubeType>::ExpandType {
        VectorTile::__expand_start_of(scope, line, space, self)
    }

    pub fn __expand_index_method(
        &self,
        scope: &Scope,
        coords: &<Coords<u32> as CubeType>::ExpandType,
        space: &Space,
    ) -> <CoordsDyn as CubeType>::ExpandType {
        VectorTile::__expand_index_of(scope, coords, space, self)
    }
}

comptime_only!(VectorTile);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::storage::{StorageLevels, StoragePartitioning};

    const K: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);

    /// A run reaches its width along its dimension and one along every other.
    #[test]
    fn a_run_reaches_its_width_along_its_dimension_alone() {
        const WIDTH: usize = 4;
        let load = VectorTile::run(N, WIDTH);
        assert_eq!(load.extent_along(N), WIDTH);
        assert_eq!(load.extent_along(K), 1);
    }

    /// A load over stored tiles reaches along each dimension what its fused tiles hold there.
    #[test]
    fn a_tiled_load_reaches_its_fused_extents() {
        let load = VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32).unwrap();
        assert_eq!(load.extent_along(K), 16);
        assert_eq!(load.extent_along(N), 2);
    }

    /// A buffer stored in no tiles counts its loads with the last dimension fastest, as it always has:
    /// a row-major operand reads exactly as before.
    #[test]
    fn a_plain_buffer_counts_its_loads_row_major() {
        let load = VectorTile::run(N, 4);
        let space = Space::new(&[(K, 8), (N, 16)]);
        assert_eq!(load.order(&space), vec![(0, 8, 1), (1, 4, 1)]);
        assert_eq!(load.offset_in(5, &space, N), 4);
        assert_eq!(load.offset_in(5, &space, K), 1);
    }

    /// A buffer stored in tiles counts its loads tile by tile: neighbouring lines are neighbouring
    /// loads of one stored tile, so the units that take neighbouring lines read one run of memory.
    #[test]
    fn a_tiled_buffer_counts_its_loads_a_stored_tile_at_a_time() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 4), (N, 8)];
        let load = VectorTile::new(&stored, N, 32).unwrap();
        let space = Space::new(&[(N, 32), (K, 128)]);
        // Loads along K inside a tile first, then N inside it, then the tiles along K, then N.
        assert_eq!(
            load.order(&space),
            vec![(0, 2, 8), (1, 2, 4), (0, 8, 1), (1, 4, 1)]
        );
        let at = |line| {
            (
                load.offset_in(line, &space, N),
                load.offset_in(line, &space, K),
            )
        };
        assert_eq!(at(1), (0, 16));
        assert_eq!(at(3), (0, 48));
        assert_eq!(at(4), (2, 0));
        assert_eq!(at(32), (0, 64));
    }

    /// A window that holds part of a stored tile along a dimension counts that dimension's loads in order:
    /// every load is still counted once.
    #[test]
    fn a_window_inside_a_stored_tile_counts_every_load_once() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 4), (N, 8)];
        let load = VectorTile::new(&stored, N, 32).unwrap();
        let space = Space::new(&[(N, 32), (K, 32)]);
        let order = load.order(&space);
        assert_eq!(order, vec![(0, 2, 8), (1, 2, 1), (0, 8, 1)]);
        let mut seen: Vec<_> = (0..32)
            .map(|line| {
                (
                    load.offset_in(line, &space, N),
                    load.offset_in(line, &space, K),
                )
            })
            .collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 32);
    }

    /// The value offset of the first value of each load a 32-lane plane takes on its first turn,
    /// over a buffer stored as `storage` holding a tensor of `extents`, the load of `line` starting
    /// at `at(line, dimension)` along each dimension.
    fn first_turn_offsets(
        storage: &StoragePartitioning,
        extents: &[(Dimension, usize)],
        at: impl Fn(usize, Dimension) -> usize,
    ) -> Vec<usize> {
        let geometry = storage.physical(extents).unwrap();
        let dimensions: Vec<Dimension> = extents.iter().map(|&(dimension, _)| dimension).collect();
        let labels = storage.labels(&dimensions);
        (0..32)
            .map(|line| {
                dimensions
                    .iter()
                    .map(|&dimension| {
                        // The dimension's axes, finest first, its coordinate unravelled over them.
                        let mut rest = at(line, dimension);
                        let mut offset = 0;
                        for (d, _) in labels
                            .iter()
                            .enumerate()
                            .rev()
                            .filter(|&(_, &a)| a == dimension)
                        {
                            let (count, stride) = geometry.axes().nth(d).unwrap();
                            offset += (rest % count) * stride;
                            rest /= count;
                        }
                        offset
                    })
                    .sum()
            })
            .collect()
    }

    /// The runs of memory a plane's first turn reads, as `(first byte, bytes)`, each load `bytes`
    /// long from its offset of `bits`-bit values: loads that lie back to back join one run.
    fn runs(offsets: &[usize], bits: usize, bytes: usize) -> Vec<(usize, usize)> {
        let mut starts: Vec<usize> = offsets.iter().map(|&offset| offset * bits / 8).collect();
        starts.sort();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for start in starts {
            match runs.last_mut() {
                Some((first, length)) if *first + *length == start => *length += bytes,
                _ => runs.push((start, bytes)),
            }
        }
        runs
    }

    /// A plane copying a window of a tile-order 4-bit weight in loads of 32 values reads, on its
    /// first turn, one stored tile: one 512-byte run. Counted with the last dimension fastest instead,
    /// the same bytes come from two tiles, two runs 512 bytes apart: the same transactions, in an
    /// order that scatters a cube's reads across memory.
    #[test]
    fn a_plane_reads_a_stored_tile_as_one_run() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 4), (N, 8)];
        let storage = StorageLevels::new(&[(K, 8)])
            .tile(&[(K, 2), (N, 2)])
            .tile(&[(K, 4), (N, 8)])
            .grid(&[K, N]);
        let extents = [(N, 64), (K, 256)];
        let window = Space::new(&[(N, 32), (K, 128)]);
        let load = VectorTile::new(&stored, N, 32).unwrap();
        let in_storage_order = first_turn_offsets(&storage, &extents, |line, dimension| {
            load.offset_in(line, &window, dimension)
        });
        let counts = load.counts(&window);
        let row_major = first_turn_offsets(&storage, &extents, |line, dimension| {
            let along = window.position(dimension);
            let inner: usize = counts[along + 1..].iter().product();
            (line / inner) % counts[along] * load.extent_along(dimension)
        });
        assert_eq!(runs(&in_storage_order, 4, 16), vec![(0, 512)]);
        assert_eq!(runs(&row_major, 4, 16), vec![(0, 256), (512, 256)]);
    }

    /// A plain buffer has no stored tile, and its transaction is the loads along its innermost dimension
    /// that fill 128 bytes: eight loads of four 32-bit values.
    #[test]
    fn a_plain_buffer_fills_a_transaction_along_its_innermost_axis() {
        let load = VectorTile::run(N, 4);
        let space = Space::new(&[(K, 8), (N, 64)]);
        assert_eq!(
            load.run_loads(Run::Load, &space, 32, 128),
            Some(vec![(K, 1), (N, 1)])
        );
        assert_eq!(
            load.run_loads(Run::Transaction, &space, 32, 128),
            Some(vec![(K, 1), (N, 8)])
        );
        assert_eq!(load.run_loads(Run::StoredTile, &space, 32, 128), None);
    }

    /// A tile-order buffer's runs follow its storage: a transaction is the stored pieces above one
    /// load until 128 bytes, a stored tile all of them, each counted along the dimension it lies on.
    #[test]
    fn a_tiled_buffer_runs_through_its_stored_pieces() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 4), (N, 8)];
        let load = VectorTile::new(&stored, N, 32).unwrap();
        let space = Space::new(&[(N, 32), (K, 128)]);
        // A load is 16 bytes of 4-bit values: four along K make 64, two columns of those 128,
        // and the whole tile's eight columns 512.
        assert_eq!(
            load.run_loads(Run::Transaction, &space, 4, 128),
            Some(vec![(N, 2), (K, 4)])
        );
        assert_eq!(
            load.run_loads(Run::StoredTile, &space, 4, 128),
            Some(vec![(N, 8), (K, 4)])
        );
        assert_eq!(
            load.run_loads(Run::Transaction, &space, 4, 64),
            Some(vec![(N, 1), (K, 4)])
        );
    }

    /// A plain buffer loads a run along its innermost axis, whatever the width.
    #[test]
    fn a_plain_buffer_loads_along_its_innermost_dimension() {
        assert_eq!(VectorTile::new(&[], N, 4).unwrap().extents(), &[(N, 4)]);
        assert_eq!(VectorTile::new(&[], N, 1).unwrap().extents(), &[(N, 1)]);
    }

    /// An NVFP4 load of four words, two along `K` and two columns, is a 16 by 2 rectangle; two
    /// words are one column's 16 values, and one word its 8.
    #[test]
    fn a_stored_buffer_loads_its_finest_tiles() {
        let stored = [(K, 8), (K, 2), (N, 2), (K, 32), (N, 8)];
        assert_eq!(
            VectorTile::new(&stored, N, 32).unwrap().extents(),
            &[(K, 16), (N, 2)]
        );
        assert_eq!(
            VectorTile::new(&stored, N, 16).unwrap().extents(),
            &[(K, 16)]
        );
        assert_eq!(VectorTile::new(&stored, N, 8).unwrap().extents(), &[(K, 8)]);
        assert_eq!(VectorTile::new(&stored, N, 32).unwrap().values(), 32);
    }

    /// A load never cuts a stored tile: four values of an eight-value word, or 24 values that would
    /// take one and a half of the next tile, are refused. Neither may it outgrow what is stated.
    #[test]
    fn a_load_that_cuts_a_stored_tile_is_refused() {
        let stored = [(K, 8), (N, 2)];
        assert_eq!(
            VectorTile::new(&stored, N, 4),
            Err(TileMisfit::Overshoots {
                dimension: K,
                wanted: 4,
                piece: 8
            })
        );
        assert!(VectorTile::new(&stored, N, 24).is_err());
        assert!(VectorTile::new(&stored, N, 32).is_err());
    }

    /// A load is one extent per dimension: stored tiles that go along `K`, then `N`, then `K` again
    /// make no load that spans all three, since its values along `K` would not be one run.
    #[test]
    fn a_load_that_returns_to_a_dimension_is_refused() {
        assert_eq!(
            VectorTile::new(&[(K, 8), (N, 2), (K, 2)], N, 32),
            Err(TileMisfit::Interleaved {
                wanted: K,
                found: N
            })
        );
    }

    /// A value's place in a load reads off its position unravelled over the load's extents, finest
    /// first: the 21st value of a 16 by 2 load is the second column's fifth, a run's is itself.
    #[test]
    fn a_position_in_a_load_is_an_offset_along_each_dimension() {
        let load = VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32).unwrap();
        assert_eq!(load.offset_along(20, K), 4);
        assert_eq!(load.offset_along(20, N), 1);
        assert_eq!(VectorTile::run(N, 4).offset_along(3, N), 3);
        assert_eq!(VectorTile::run(N, 4).offset_along(3, K), 0);
    }

    /// A stated tile of one is an axis of the buffer, not a piece of a load.
    #[test]
    fn a_stored_tile_of_one_is_skipped() {
        let stored = [(N, 4), (N, 1), (K, 2)];
        assert_eq!(
            VectorTile::new(&stored, N, 8).unwrap().extents(),
            &[(N, 4), (K, 2)]
        );
    }

    /// A reader that places a load along one dimension refuses a rectangle, naming itself, rather
    /// than landing half its values on the wrong column.
    #[test]
    #[should_panic(expected = "the fragment load places a load along one dimension")]
    fn a_rectangle_is_refused_by_a_one_dimension_reader() {
        let tile = VectorTile::new(&[(K, 8), (N, 2)], N, 16).unwrap();
        tile.along_one_dimension("the fragment load");
    }

    /// Read in loads, a plain buffer's innermost axis counts lines and nothing else moves; an
    /// NVFP4 buffer's word, its second word along `K` and its second column are each one load's
    /// whole, wherever the level-major order puts them, and a batch axis ahead is untouched.
    #[test]
    fn a_load_covers_the_innermost_axis_or_the_stored_tiles_it_takes() {
        let run = VectorTile::run(N, 4);
        assert_eq!(run.parts(&[], &[K, N], 2), vec![1, 4]);

        // `[m, k]` stored as a word of 8 along `K`, 2 words along `K` by 2 rows, then 4 by 4:
        // `[M, K, M, K, M, K, K]` level-major, the word the last axis.
        const M: Dimension = Dimension(2);
        let stored = [(K, 8), (K, 2), (M, 2), (K, 4), (M, 4)];
        let labels = [M, K, M, K, M, K, K];
        let load = VectorTile::new(&stored, K, 32).unwrap();
        assert_eq!(load.extents(), &[(K, 16), (M, 2)]);
        assert_eq!(load.parts(&stored, &labels, 7), vec![1, 1, 1, 1, 2, 2, 8]);
        assert_eq!(
            load.parts(&stored, &labels, 8),
            vec![1, 1, 1, 1, 1, 2, 2, 8]
        );
    }

    /// Every stated tile's axis reads at the tile's count, and only the grid axes are left to the
    /// buffer's shape: a read over the tiles then divides by constants. A batch axis ahead is
    /// the buffer's too.
    #[test]
    fn the_stated_tiles_have_their_stated_extents() {
        const M: Dimension = Dimension(2);
        let stored = [(K, 8), (K, 2), (M, 2), (K, 4), (M, 4)];
        let labels = [M, K, M, K, M, K, K];
        let tiled = [Some(4), Some(4), Some(2), Some(2), Some(8)];
        let expected: Vec<_> = [None, None].into_iter().chain(tiled).collect();
        assert_eq!(VectorTile::stated_extents(&stored, &labels, 7), expected);
        let batched: Vec<_> = [None].into_iter().chain(expected).collect();
        assert_eq!(VectorTile::stated_extents(&stored, &labels, 8), batched);
    }
}

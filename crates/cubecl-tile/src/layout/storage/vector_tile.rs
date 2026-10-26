//! [`VectorTile`]: the values one vector load brings, as a tile.
//!
//! An operand says how it is stored, and the kernel picks how wide it loads; the tile a load
//! covers follows from both. A plain buffer is loaded in a run along its innermost axis. A buffer
//! stored in tiles is loaded a stored tile at a time: its finest tiles, fused per axis, until they
//! hold the load's values, so an NVFP4 buffer whose word runs along `K` and whose next pieces are
//! two words along `K` and two columns loads 32 values as 16 along `K` by 2 along `N`.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::{
    Axis, Coords, Extent, Space, TileMisfit,
    algebra::{Integer, IntegerExpand},
    comptime_only,
};

/// The values one vector load brings, as a tile: its extents, finest first, one per axis.
///
/// ```ignore
/// // a plain row-major buffer, four wide
/// VectorTile::new(&[], N, 4)                                  // [(N, 4)]
/// // NVFP4 values: a word of 8 along K, 2 words along K, 2 columns; 32 values a load
/// VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32)           // [(K, 16), (N, 2)]
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VectorTile {
    extents: Vec<(Axis, usize)>,
}

impl VectorTile {
    /// The tile a load of `values` values covers over a buffer stored in `stored` tiles, finest
    /// first ([`TileSpec::stored_tiles`](crate::TileSpec::stored_tiles)): its finest tiles, fused
    /// where one axis continues, until they hold `values`. Nothing stated, a run of `values`
    /// along `innermost`.
    ///
    /// # Errors
    ///
    /// A load that would cut a stored tile, one wider than every stored tile together, or one
    /// whose axis comes back after another, which is no longer one extent per axis.
    pub fn new(
        stored: &[(Axis, usize)],
        innermost: Axis,
        values: usize,
    ) -> Result<Self, TileMisfit> {
        if stored.is_empty() || values == 1 {
            return Ok(Self::run(innermost, values));
        }
        let mut extents: Vec<(Axis, usize)> = Vec::new();
        let mut held = 1;
        for &(axis, count) in stored.iter().filter(|&&(_, count)| count > 1) {
            if held == values {
                break;
            }
            if !(values / held).is_multiple_of(count) {
                return Err(TileMisfit::Overshoots {
                    axis,
                    wanted: values / held,
                    piece: count,
                });
            }
            held *= count;
            match extents.last().copied() {
                Some((last, extent)) if last == axis => {
                    *extents.last_mut().expect("just read") = (axis, extent * count);
                }
                Some((last, _)) if extents.iter().any(|&(a, _)| a == axis) => {
                    return Err(TileMisfit::Interleaved {
                        wanted: axis,
                        found: last,
                    });
                }
                _ => extents.push((axis, count)),
            }
        }
        if held != values {
            return Err(TileMisfit::RunsOut {
                wanted: extents.last().map_or(innermost, |&(axis, _)| axis),
            });
        }
        Ok(Self { extents })
    }

    /// A run of `values` along `axis`: how a plain buffer is loaded.
    pub fn run(axis: Axis, values: usize) -> Self {
        Self {
            extents: vec![(axis, values)],
        }
    }

    /// The extents, finest first.
    pub fn extents(&self) -> &[(Axis, usize)] {
        &self.extents
    }

    /// Values one load holds along its finest axis: one line of it, the whole load where it spans
    /// one axis. A reader walking lines along that axis takes `values() / run()` of them a load.
    pub(crate) fn run_length(&self) -> usize {
        self.extents.first().map_or(1, |&(_, run)| run)
    }

    /// The box one load covers, an axis to each extent.
    pub fn space(&self) -> Space {
        Space::new(&self.extents)
    }

    /// How many values the load brings.
    pub fn values(&self) -> usize {
        self.extents.iter().map(|&(_, extent)| extent).product()
    }

    /// The axis a one-axis load runs along and its width, for a `reader` that places a load along
    /// one axis.
    ///
    /// # Panics
    ///
    /// When the load spans several axes, naming `reader`.
    pub(crate) fn along_one_axis(&self, reader: &str) -> (Axis, usize) {
        match self.extents.as_slice() {
            &[one] => one,
            extents => panic!(
                "{reader} places a load along one axis, and this operand loads {extents:?}: \
                 read it with a narrower vector, or through a path that places a tile"
            ),
        }
    }

    /// How far one load reaches along `axis`: its extent there, `1` off the axes it spans.
    pub(crate) fn extent_along(&self, axis: Axis) -> usize {
        self.extents
            .iter()
            .find(|&&(a, _)| a == axis)
            .map_or(1, |&(_, extent)| extent)
    }

    /// How many values of one load lie between two neighbours along `axis`: the product of the
    /// extents finer than `axis`'s. `0` off the axes the load spans.
    pub(crate) fn step_along(&self, axis: Axis) -> usize {
        let mut step = 1;
        for &(a, extent) in &self.extents {
            if a == axis {
                return step;
            }
            step *= extent;
        }
        0
    }

    /// How far along `axis` the value at `position` of one load sits from the load's first: the
    /// position unravelled over the load's extents, finest first. `20` of a 16 by 2 load along `K`
    /// then `N` is 4 along `K` and 1 along `N`.
    pub(crate) fn offset_along(&self, position: usize, axis: Axis) -> usize {
        let mut rest = position;
        for &(a, extent) in &self.extents {
            if a == axis {
                return rest % extent;
            }
            rest /= extent;
        }
        0
    }

    /// How many loads an edge of `edge` values along `axis` holds, rounded up: an edge cuts whole
    /// loads, or is the axis's whole `extent`, since a padded stage's innermost extent need not
    /// fill its last load.
    ///
    /// # Panics
    ///
    /// An edge that is neither, so the next region would start mid-load. A `Dynamic` axis has no
    /// extent to be cut whole, so it owes the divisibility.
    pub(crate) fn loads_in(&self, axis: Axis, edge: usize, extent: Extent) -> usize {
        let w = self.extent_along(axis);
        assert!(
            edge.is_multiple_of(w) || matches!(extent, Extent::Static(x) if x == edge),
            "Memory::at: the edge {edge} along {axis:?} is neither a whole number of loads {w} \
             wide there nor the axis's whole extent ({extent:?}), so a step would start mid-load"
        );
        edge.div_ceil(w)
    }

    /// How many loads fit along each axis of `space`, rounded up: a padded stage's innermost extent
    /// need not fill whole loads, and a checked read's box must include the partial last one.
    pub(crate) fn counts(&self, space: &Space) -> Vec<usize> {
        space
            .axes()
            .map(|axis| space.extent(axis).div_ceil(self.extent_along(axis)))
            .collect()
    }

    /// How much of each of a buffer's `rank` dims one load covers, over a buffer stored in
    /// `stored` tiles, finest first, whose trailing dims `labels` name: a stated tile the load
    /// takes whole is covered entirely, and a plain buffer's innermost dim by the load's width.
    /// `1` elsewhere.
    ///
    /// A buffer read in loads is the same buffer with each dim's extent divided by what one load
    /// covers of it and every stride counted in loads: `extent / part` and `stride * part /
    /// values`, which for a plain buffer is its innermost dim in lines and the rest's strides
    /// divided by the width.
    pub(crate) fn parts(
        &self,
        stored: &[(Axis, usize)],
        labels: &[Axis],
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
        for (&(_, count), dim) in stored.iter().zip(Self::stated_dims(stored, labels, rank)) {
            if count == 1 {
                continue;
            }
            if taken == self.values() {
                break;
            }
            parts[dim] = count;
            taken *= count;
        }
        parts
    }

    /// The extent of each of a buffer's `rank` dims that is a stated tile, over a buffer stored
    /// in `stored` tiles whose trailing dims `labels` name; `None` for a dim the buffer's own
    /// shape sets. A stated tile's extent is fixed where the kernel is written, so a read over it
    /// divides by a constant rather than by a dim read off the buffer.
    pub(crate) fn stated_extents(
        stored: &[(Axis, usize)],
        labels: &[Axis],
        rank: usize,
    ) -> Vec<Option<usize>> {
        let mut extents = vec![None; rank];
        for (&(_, count), dim) in stored.iter().zip(Self::stated_dims(stored, labels, rank)) {
            extents[dim] = Some(count);
        }
        extents
    }

    /// The dim of a buffer of `rank` dims, whose trailing dims `labels` name, each of `stored`'s
    /// tiles is: the `j`-th finest stated tile of an axis is its `j`-th dim from the end.
    fn stated_dims(stored: &[(Axis, usize)], labels: &[Axis], rank: usize) -> Vec<usize> {
        let unlabelled = rank - labels.len();
        let mut from_end: Vec<(Axis, usize)> = Vec::new();
        stored
            .iter()
            .map(|&(axis, _)| {
                let nth = match from_end.iter_mut().find(|(a, _)| *a == axis) {
                    Some((_, seen)) => {
                        *seen += 1;
                        *seen
                    }
                    None => {
                        from_end.push((axis, 1));
                        1
                    }
                };
                let dim = (0..labels.len())
                    .rev()
                    .filter(|&d| labels[d] == axis)
                    .nth(nth - 1)
                    .expect("a stated tile is a dim of the buffer");
                unlabelled + dim
            })
            .collect()
    }
}

#[cube]
impl VectorTile {
    fn count_of(#[comptime] space: &Space, #[comptime] tile: &VectorTile) -> u32 {
        comptime!(tile.counts(space).iter().product::<usize>() as u32).runtime()
    }

    fn start_of(
        line: u32,
        #[comptime] space: &Space,
        #[comptime] tile: &VectorTile,
    ) -> Coords<u32> {
        let rank = comptime!(space.rank());
        let place = Coords::constant(comptime!(tile.counts(space))).unravel(line);
        let mut coords = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let extent = comptime!(tile.extent_along(space.axis_at(p)) as u32);
            coords.push(place.at(p).times(extent));
        }
        coords
    }

    fn index_of(
        coords: &Coords<u32>,
        #[comptime] space: &Space,
        #[comptime] tile: &VectorTile,
    ) -> CoordsDyn {
        let rank = comptime!(space.rank());
        let mut index = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            let extent = comptime!(tile.extent_along(space.axis_at(p)) as u32);
            index.push(coords.at(p).divided_by(extent));
        }
        index
    }
}

/// `load.count(&space)`, `load.start(line, &space)`, `load.index(&coords, &space)`: written out by
/// hand, since a compile-time value has no kernel-side twin for `#[cube]` to hang a method on.
impl VectorTile {
    /// How many loads a window spanning `space` holds.
    pub fn count(&self, space: &Space) -> u32 {
        VectorTile::count_of(space, self)
    }

    /// The first value of the `line`-th load of a window spanning `space`, one coordinate per
    /// axis, the loads counted with the last axis fastest.
    pub fn start(&self, line: u32, space: &Space) -> Coords<u32> {
        VectorTile::start_of(line, space, self)
    }

    /// Where the load starting at `coords` sits in a view of a window spanning `space` read in
    /// these loads: each coordinate divided by the load's extent along its axis.
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

    const K: Axis = Axis(0);
    const N: Axis = Axis(1);

    /// A plain buffer loads a run along its innermost axis, whatever the width.
    #[test]
    fn a_plain_buffer_loads_along_its_innermost_axis() {
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
                axis: K,
                wanted: 4,
                piece: 8
            })
        );
        assert!(VectorTile::new(&stored, N, 24).is_err());
        assert!(VectorTile::new(&stored, N, 32).is_err());
    }

    /// A load is one extent per axis: stored tiles that go along `K`, then `N`, then `K` again
    /// make no load that spans all three, since its values along `K` would not be one run.
    #[test]
    fn a_load_that_returns_to_an_axis_is_refused() {
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
    fn a_position_in_a_load_is_an_offset_along_each_axis() {
        let load = VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32).unwrap();
        assert_eq!(load.offset_along(20, K), 4);
        assert_eq!(load.offset_along(20, N), 1);
        assert_eq!(VectorTile::run(N, 4).offset_along(3, N), 3);
        assert_eq!(VectorTile::run(N, 4).offset_along(3, K), 0);
    }

    /// A stated tile of one is a dim of the buffer, not a piece of a load.
    #[test]
    fn a_stored_tile_of_one_is_skipped() {
        let stored = [(N, 4), (N, 1), (K, 2)];
        assert_eq!(
            VectorTile::new(&stored, N, 8).unwrap().extents(),
            &[(N, 4), (K, 2)]
        );
    }

    /// A reader that places a load along one axis refuses a rectangle, naming itself, rather
    /// than landing half its values on the wrong column.
    #[test]
    #[should_panic(expected = "the fragment load places a load along one axis")]
    fn a_rectangle_is_refused_by_a_one_axis_reader() {
        let tile = VectorTile::new(&[(K, 8), (N, 2)], N, 16).unwrap();
        tile.along_one_axis("the fragment load");
    }

    /// Read in loads, a plain buffer's innermost dim counts lines and nothing else moves; an
    /// NVFP4 buffer's word, its second word along `K` and its second column are each one load's
    /// whole, wherever the level-major order puts them, and a batch dim ahead is untouched.
    #[test]
    fn a_load_covers_the_innermost_dim_or_the_stored_tiles_it_takes() {
        let run = VectorTile::run(N, 4);
        assert_eq!(run.parts(&[], &[K, N], 2), vec![1, 4]);

        // `[m, k]` stored as a word of 8 along `K`, 2 words along `K` by 2 rows, then 4 by 4:
        // `[M, K, M, K, M, K, K]` level-major, the word the last dim.
        const M: Axis = Axis(2);
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

    /// Every stated tile's dim reads at the tile's count, and only the grid dims are left to the
    /// buffer's shape: a read over the tiles then divides by constants. A batch dim ahead is
    /// the buffer's too.
    #[test]
    fn the_stated_tiles_have_their_stated_extents() {
        const M: Axis = Axis(2);
        let stored = [(K, 8), (K, 2), (M, 2), (K, 4), (M, 4)];
        let labels = [M, K, M, K, M, K, K];
        let tiled = [Some(4), Some(4), Some(2), Some(2), Some(8)];
        let expected: Vec<_> = [None, None].into_iter().chain(tiled).collect();
        assert_eq!(VectorTile::stated_extents(&stored, &labels, 7), expected);
        let batched: Vec<_> = [None].into_iter().chain(expected).collect();
        assert_eq!(VectorTile::stated_extents(&stored, &labels, 8), batched);
    }
}

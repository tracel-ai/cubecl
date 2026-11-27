//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`, with non-fragment operands
//! loaded into transient `A`/`B` fragments.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::ops::matmul::leaf::Side;
use crate::*;

#[cube]
impl CmmaData {
    /// Tensor-core contraction `self += lhs · rhs`, each factor times its scales.
    pub(crate) fn mma<A: Numeric, EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: &Space,
    ) {
        let (a_order, b_order) = (lhs.stage_order(), rhs.stage_order());
        let a_read = comptime!(FragmentRead::new(Side::Lhs, &lhs.place.space, out, a_order));
        let b_read = comptime!(FragmentRead::new(Side::Rhs, &rhs.place.space, out, b_order));
        cmma::execute(
            &lhs.cmma_fragment::<EL>(MatrixIdent::A, a_read),
            &rhs.cmma_fragment::<ER>(MatrixIdent::B, b_read),
            &self.fragment.downcast::<A>(),
            &self.fragment.downcast::<A>(),
        );
    }
}

#[cube]
impl Tile {
    /// This factor as the `ident` fragment of a tensor-core contraction: the one it holds, or its
    /// window loaded as `read` says.
    pub(crate) fn cmma_fragment<E: Numeric>(
        &self,
        #[comptime] ident: MatrixIdent,
        #[comptime] read: FragmentRead,
    ) -> Matrix<E> {
        match &self.kind {
            TileKind::PlaneTile(tile) => match tile {
                PlaneTile::Cmma(fragment) => self.held_fragment::<E>(fragment),
                PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                    panic!("Tile::cmma_fragment: a held operand should be a cmma fragment")
                }
            },
            TileKind::Memory(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => self.loaded_fragment::<E>(ident, read),
        }
    }

    /// The fragment this factor holds, which carries no scale.
    fn held_fragment<E: Numeric>(&self, fragment: &CmmaData) -> Matrix<E> {
        self.refuse_factor("PlaneTile::Mma");
        fragment.fragment.downcast::<E>()
    }

    /// A new `ident` fragment holding this factor's window, loaded as `read` says.
    fn loaded_fragment<E: Numeric>(
        &self,
        #[comptime] ident: MatrixIdent,
        #[comptime] read: FragmentRead,
    ) -> Matrix<E> {
        let mut fragment =
            unsafe { Matrix::<E>::uninitialized(ident, read.m, read.n, read.k, read.layout) };
        self.load(&mut fragment, read);
        fragment
    }
}

/// How one factor of a tensor-core contraction is read into its fragment.
#[derive(Clone)]
pub(crate) struct FragmentRead {
    /// The fragment's edges.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// The window's layout, read off its innermost physical dimension.
    pub layout: MatrixLayout,
    /// What this factor's scales are checked against.
    pub side: Side,
    pub out: Space,
    /// The order the window lays its dimensions down in.
    pub order: StageOrder,
}

impl FragmentRead {
    /// How the `side` factor spanning `own`, laid down in `order`, is read for a contraction into
    /// `out`, from it alone.
    pub(crate) fn new(side: Side, own: &Space, out: &Space, order: StageOrder) -> Self {
        let acc_dimensions = accumulator_dimensions(side, out, own);
        let m = acc_dimensions.rows(out);
        let n = acc_dimensions.cols(out);
        // A split contraction is one `k` edge.
        let k = own.contracted_extent(out);
        // Whether the window's innermost dimension is one of the output's, its rows for the lhs and
        // its columns for the rhs, or the contracted one: a window is read as it is laid out.
        let (trailing_kept, _) = innermost_run(own, out, order);
        let layout = match (side, trailing_kept) {
            (Side::Lhs, false) | (Side::Rhs, true) => MatrixLayout::RowMajor,
            (Side::Lhs, true) | (Side::Rhs, false) => MatrixLayout::ColMajor,
        };
        FragmentRead {
            m,
            n,
            k,
            layout,
            side,
            out: out.clone(),
            order,
        }
    }
}

/// Where `window`'s innermost run lies against `out`, the window laid down in `order`: whether
/// its innermost dimension is one of `out`'s, and the last dimension laid on the other side,
/// `None` where every dimension lies on the innermost's side.
fn innermost_run(window: &Space, out: &Space, order: StageOrder) -> (bool, Option<Dimension>) {
    match order {
        StageOrder::AsSource => innermost_run_in_space_order(window, out),
        StageOrder::AlongRun {
            dimension: moved, ..
        } => innermost_run_in_space_order(&laid_along(window, moved), out),
    }
}

/// [`innermost_run`] of a window laid down in its Space's order, its last dimension innermost.
fn innermost_run_in_space_order(window: &Space, out: &Space) -> (bool, Option<Dimension>) {
    let rank = window.rank();
    let kept = |p: usize| out.contains(window.dimension_at(p));
    let innermost = kept(rank - 1);
    let other_side = (0..rank).rev().find(|&p| kept(p) != innermost);
    (innermost, other_side.map(|p| window.dimension_at(p)))
}

/// How an `ident` fragment reads a window over `space` laid down in `order` as that order lays it,
/// where it is not its Space's: the layout its lanes' runs of `k` are contiguous in, and the
/// position of the dimension its rows step along. `None` for a window in its Space's order, which
/// the fragment reads in the layout it was opened with.
pub(crate) fn laid_fragment_read(
    ident: MatrixIdent,
    space: &Space,
    order: StageOrder,
) -> Option<(MatrixLayout, usize)> {
    let StageOrder::AlongRun {
        dimension: moved, ..
    } = order
    else {
        return None;
    };
    let layout = match ident {
        MatrixIdent::A => MatrixLayout::RowMajor,
        MatrixIdent::B => MatrixLayout::ColMajor,
        MatrixIdent::Accumulator => return None,
    };
    let laid = laid_along(space, moved);
    let rows = laid.dimension_at(MatrixDimensions::edges(&laid).row_split);
    Some((layout, space.position(rows)))
}

/// `window`'s dimensions in the order a stage laid along `moved` lays them down: `moved` last.
fn laid_along(window: &Space, moved: Dimension) -> Space {
    let laid: Vec<Dimension> = window
        .dimensions()
        .filter(|&dimension| dimension != moved)
        .chain([moved])
        .collect();
    window.subspace(&laid)
}

/// Whether a factor reaches its fragment through its plane's landing rather than as it lies: a
/// scaled factor, one packed other than as stored, or one outside shared memory.
fn loads_through_landing(scaled: bool, as_stored: bool, shared: bool) -> bool {
    scaled || !as_stored || !shared
}

#[cube]
impl Tile {
    /// This factor read into `frag`. A fragment loads a shared, plain window as it lies; a scaled
    /// or packed factor, or a global one whose layout the load cannot check, goes through its
    /// landing.
    pub(crate) fn load<E: Numeric>(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let scaled = self.quantized();
        let packing = self.packing();
        let shared = self.is_shared();
        if comptime!(loads_through_landing(scaled, packing.as_stored(), shared)) {
            self.load_landed::<E>(frag, read.clone());
        } else {
            self.load_in_place::<E>(frag, read.clone());
        }
    }

    /// This factor read into `frag` through its plane's landing.
    fn load_landed<E: Numeric>(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let landing = self.landed::<E>(read.side, &read.out);
        landing.load_into(frag, &read.out);
        // The landing is this region's until every unit's load has read it.
        sync_plane();
    }

    /// This factor's shared, plain window read into `frag` as it lies.
    fn load_in_place<E: Numeric>(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        // A row is every dimension past the last one on the other side of the output, which a
        // window of several dimensions a side holds as one run: its stride is that dimension's.
        let row = comptime!({
            let space = &self.place.space;
            let (_, other_side) = innermost_run(space, &read.out, read.order);
            other_side.map_or(space.rank() - 2, |dimension| space.position(dimension))
        });
        match &self.kind {
            TileKind::Memory(m) => cmma::load(frag, m.window_slice::<E>(), m.row_stride_at(row)),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("mma: an operand should reach a fragment from a memory window")
            }
        }
    }

    /// This factor as `instruction` reads it over its whole window: in its plane's landing where
    /// the tensor cores read it through one, scaled or packed, and as it is otherwise.
    pub(crate) fn landed_for<S: Numeric>(
        &self,
        #[comptime] side: Side,
        #[comptime] out: &Space,
        #[comptime] instruction: Instruction,
    ) -> Tile {
        let scaled = self.quantized();
        let packing = self.packing();
        if comptime!(instruction == Instruction::Cmma && (scaled || !packing.as_stored())) {
            self.landed::<S>(side, out)
        } else {
            self.clone()
        }
    }

    /// This factor in its plane's landing: a dense shared-memory stage holding `values ⊗ scales`
    /// that fragments load from.
    pub(crate) fn landed<S: Numeric>(
        &self,
        #[comptime] side: Side,
        #[comptime] out: &Space,
    ) -> Tile {
        let space = comptime!(self.place.space.clone());
        let units = self.units();
        let planes = comptime!(plane_windows(&space, &self.place.levels));
        let (stage, mut window) = Memory::landing::<S>(&space, units, planes);
        let landing = Tile::new::<S>(stage.kind, comptime!(self.place.clone()));

        // A load is read whole, a tile of several dimensions included, and landed a run of its
        // finest dimension at a time, each run under the scale at its own coordinates.
        let load = self.vector_tile();
        let (sw, (finest, run)) = comptime!((load.values(), load.extents()[0]));
        let size!(SW) = sw;
        let size!(RUN) = run;
        let view = self.nd_packed::<S, SW>(comptime!(Guard::Checked));
        let acc_dimensions = comptime!(accumulator_dimensions(side, out, &space));
        let scales = self.reader(
            comptime!(MatrixDimensions::trailing(&space)),
            0usize,
            side,
            out.clone(),
            acc_dimensions,
        );
        let lines = load.count(&space);
        let rank = space.rank();
        let strides = comptime!(dense_strides(&space));
        let run_stride = comptime!(strides[space.position(finest)]);
        // Every unit joins every turn, so a scale read by shuffle has the whole plane; units past
        // the lines reread the last one and write nothing.
        #[allow(clippy::manual_div_ceil)]
        let turns = (lines + PLANE_DIM - 1) / PLANE_DIM;
        for turn in 0..turns {
            let mine = turn * PLANE_DIM + UNIT_POS_PLANE;
            let start = load.start(min(mine, lines - 1), &space);
            let held = view.read(load.index(&start, &space));
            #[unroll]
            for r in 0..(sw / run) {
                let offset = r * run;
                let mut at = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    at.push(
                        start.at(p)
                            + comptime!(load.offset_along(offset, space.dimension_at(p)) as u32),
                    );
                }
                let mut values = Vector::<S, RUN>::empty();
                #[unroll]
                for v in 0..run {
                    values.insert(v, held.extract(offset + v));
                }
                let landed = scales.apply_at::<S, RUN>(values, &at);
                if mine < lines {
                    let base = offset_of(&at, strides.clone());
                    #[unroll]
                    for v in 0..run {
                        window[base + v * run_stride] = landed.extract(v);
                    }
                }
            }
        }
        sync_plane();
        landing
    }
}

#[cube]
impl Tile {
    /// This landing loaded into `frag`, dense over the window's dimensions.
    pub(crate) fn load_into<E: Numeric>(&self, frag: &mut Matrix<E>, #[comptime] out: &Space) {
        let cols = comptime!(landed_cols(&self.place.space, out) as u32);
        match &self.kind {
            TileKind::Memory(m) => {
                comptime!(assert!(
                    m.address == AddressSpace::Shared,
                    "mma: a fragment loads from a shared window"
                ));
                cmma::load(frag, m.window_slice::<E>(), cols)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!("mma: a fragment loads from a shared window"),
        }
    }
}

/// The column count of a landed window's matrix: the trailing dimensions on the innermost
/// dimension's side of the contraction. A landing is laid down in its Space's order.
fn landed_cols(window: &Space, out: &Space) -> usize {
    let (_, other_side) = innermost_run(window, out, StageOrder::AsSource);
    let first = other_side.map_or(0, |dimension| window.position(dimension) + 1);
    (first..window.rank())
        .map(|p| window.extent_at(p))
        .product()
}

/// The accumulator's [`MatrixDimensions`] read off one factor alone.
fn accumulator_dimensions(side: Side, out: &Space, own: &Space) -> MatrixDimensions {
    let mut col_split = out.rank() - 1;
    let in_columns = |dimension: Dimension| match side {
        Side::Lhs => !own.contains(dimension),
        Side::Rhs => own.contains(dimension),
    };
    while col_split > 1 && in_columns(out.dimension_at(col_split - 1)) {
        col_split -= 1;
    }
    MatrixDimensions {
        row_split: col_split - 1,
        col_split,
    }
}

/// Row-major scalar strides over a dense copy of `space`.
fn dense_strides(space: &Space) -> Vec<usize> {
    let rank = space.rank();
    let mut strides = vec![1; rank];
    for p in (0..rank - 1).rev() {
        let below = strides[p + 1] * space.extent_at(p + 1);
        strides[p] = below;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    fn out() -> Space {
        Space::new(&[(M, 16), (N, 16)])
    }

    /// A window whose innermost dimension is the contracted one reads its lhs row-major and its
    /// rhs column-major.
    #[test]
    fn a_window_contracted_innermost_reads_lhs_by_rows_and_rhs_by_columns() {
        let lhs = Space::new(&[(M, 16), (K, 16)]);
        let rhs = Space::new(&[(N, 16), (K, 16)]);
        let order = StageOrder::AsSource;
        let lhs_read = FragmentRead::new(Side::Lhs, &lhs, &out(), order);
        let rhs_read = FragmentRead::new(Side::Rhs, &rhs, &out(), order);
        assert_eq!(lhs_read.layout, MatrixLayout::RowMajor);
        assert_eq!(rhs_read.layout, MatrixLayout::ColMajor);
    }

    /// A window whose innermost dimension is one of the output's reads its lhs column-major and
    /// its rhs row-major.
    #[test]
    fn a_window_kept_innermost_reads_lhs_by_columns_and_rhs_by_rows() {
        let lhs = Space::new(&[(K, 16), (M, 16)]);
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let order = StageOrder::AsSource;
        let lhs_read = FragmentRead::new(Side::Lhs, &lhs, &out(), order);
        let rhs_read = FragmentRead::new(Side::Rhs, &rhs, &out(), order);
        assert_eq!(lhs_read.layout, MatrixLayout::ColMajor);
        assert_eq!(rhs_read.layout, MatrixLayout::RowMajor);
    }

    /// The run sharing the innermost dimension's side stops at the last dimension on the other
    /// side: a split column group is one run.
    #[test]
    fn the_innermost_run_stops_at_the_other_side() {
        let (n0, n1) = (Dimension(3), Dimension(4));
        let out = Space::new(&[(M, 16), (n0, 2), (n1, 8)]);
        let rhs = Space::new(&[(K, 16), (n0, 2), (n1, 8)]);
        assert_eq!(
            innermost_run(&rhs, &out, StageOrder::AsSource),
            (true, Some(K))
        );
        assert_eq!(landed_cols(&rhs, &out), 16);
    }

    /// A window lying wholly on one side has no dimension on the other, and its whole extent is
    /// one run.
    #[test]
    fn a_window_on_one_side_is_one_run() {
        let rhs = Space::new(&[(N, 4), (Dimension(3), 4)]);
        let out = Space::new(&[(M, 16), (N, 4), (Dimension(3), 4)]);
        assert_eq!(
            innermost_run(&rhs, &out, StageOrder::AsSource),
            (true, None)
        );
        assert_eq!(landed_cols(&rhs, &out), 16);
    }

    /// A stage laid along its run of `k` reads as its physical order lays it, whatever order its
    /// Space lists: a row-major rhs laid `k` innermost reads column-major, its rows stepping
    /// along `n`.
    #[test]
    fn fragment_read_takes_the_physical_order() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let order = StageOrder::AlongRun {
            dimension: K,
            run: 8,
        };
        let read = FragmentRead::new(Side::Rhs, &rhs, &out(), order);
        assert_eq!(read.layout, MatrixLayout::ColMajor);
        assert_eq!(innermost_run(&rhs, &out(), order), (false, Some(N)));
    }

    /// A fragment opened in its Space's order reads a window laid along `k` as the stage lays it:
    /// an rhs `[k, n]` column-major, its rows stepping along `n`.
    #[test]
    fn a_fragment_reads_a_window_laid_along_k_as_it_lies() {
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        let laid = laid_fragment_read(
            MatrixIdent::B,
            &rhs,
            StageOrder::AlongRun {
                dimension: K,
                run: 8,
            },
        );
        assert_eq!(laid, Some((MatrixLayout::ColMajor, 1)));
        let lhs = Space::new(&[(K, 16), (M, 16)]);
        let laid = laid_fragment_read(
            MatrixIdent::A,
            &lhs,
            StageOrder::AlongRun {
                dimension: K,
                run: 8,
            },
        );
        assert_eq!(laid, Some((MatrixLayout::RowMajor, 1)));
        assert_eq!(
            laid_fragment_read(MatrixIdent::B, &rhs, StageOrder::AsSource),
            None
        );
    }

    /// A scaled, repacked or global factor lands first; a plain shared one loads as it lies.
    #[test]
    fn only_a_plain_shared_factor_loads_as_it_lies() {
        assert!(!loads_through_landing(false, true, true));
        assert!(loads_through_landing(true, true, true));
        assert!(loads_through_landing(false, false, true));
        assert!(loads_through_landing(false, true, false));
    }
}

/// The scalar offset of `coords` under `strides`.
#[cube]
#[allow(clippy::needless_range_loop)]
fn offset_of(coords: &Coords<u32>, #[comptime] strides: Vec<usize>) -> usize {
    let n = coords.len();
    let mut offset = 0u32.runtime();
    #[unroll]
    for p in 0..n {
        offset = offset.plus(coords.at(p).times(comptime!(strides[p] as u32)));
    }
    offset as usize
}

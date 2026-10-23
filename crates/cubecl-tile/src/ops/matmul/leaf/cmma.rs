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
        #[comptime] out: Space,
    ) {
        let a_read = comptime!(FragmentRead::new(Side::Lhs, &lhs.place.space, &out));
        let b_read = comptime!(FragmentRead::new(Side::Rhs, &rhs.place.space, &out));
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
                PlaneTile::Cmma(fragment) => {
                    self.refuse_factor("PlaneTile::Mma");
                    fragment.fragment.downcast::<E>()
                }
                PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                    panic!("Tile::cmma_fragment: a held operand must be a cmma fragment")
                }
            },
            TileKind::Memory(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                let mut fragment = unsafe {
                    Matrix::<E>::uninitialized(
                        ident,
                        read.m,
                        read.n,
                        read.k,
                        comptime!(read.layout),
                    )
                };
                self.load(&mut fragment, comptime!(read));
                fragment
            }
        }
    }
}

/// How a rhs window is read: col-major when its trailing axis is `contracted`, else row-major.
pub(crate) fn rhs_layout(rhs: &Space, contracted: Axis) -> MatrixLayout {
    match rhs.axis_at(rhs.rank() - 1) == contracted {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    }
}

/// How one factor of a tensor-core contraction is read into its fragment.
#[derive(Clone)]
pub(crate) struct FragmentRead {
    /// The fragment's edges.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// The rhs window's layout; the lhs is always row-major.
    pub layout: MatrixLayout,
    /// What this factor's scales are checked against.
    pub side: Side,
    pub out: Space,
}

impl FragmentRead {
    /// How the `side` factor spanning `own` is read for a contraction into `out`, from it alone.
    pub(crate) fn new(side: Side, own: &Space, out: &Space) -> Self {
        let acc_axes = accumulator_axes(side, out, own);
        let m = acc_axes.rows(out);
        let n = acc_axes.cols(out);
        // A split contraction is one `k` edge.
        let k = own.contracted_extent(out);
        // The lhs is row-major; the rhs col-major where its trailing axis is contracted.
        let layout = match side {
            Side::Lhs => MatrixLayout::RowMajor,
            Side::Rhs if out.contains(own.axis_at(own.rank() - 1)) => MatrixLayout::RowMajor,
            Side::Rhs => MatrixLayout::ColMajor,
        };
        FragmentRead {
            m,
            n,
            k,
            layout,
            side,
            out: out.clone(),
        }
    }
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
        if comptime!(scaled || packing != Packing::Plain || !shared) {
            let landing = self.landed::<E>(comptime!(read.side), comptime!(read.out.clone()));
            landing.load_into(frag, comptime!(read.out.clone()));
            // The landing is this region's until every unit's load has read it.
            sync_plane();
        } else {
            // A row is every axis past the last one on the other side of the output, which a
            // window of several axes a side holds as one run: its stride is that axis's.
            let row = comptime!({
                let space = &self.place.space;
                let rank = space.rank();
                let innermost = read.out.contains(space.axis_at(rank - 1));
                (0..rank)
                    .rev()
                    .find(|&p| read.out.contains(space.axis_at(p)) != innermost)
                    .unwrap_or(rank - 2)
            });
            match &self.kind {
                TileKind::Memory(m) => {
                    cmma::load(frag, m.window_slice::<E>(), m.row_stride_at(row))
                }
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("mma: an operand reaches a fragment from a memory window")
                }
            }
        }
    }

    /// This factor as `instruction` reads it over its whole window: in its plane's landing where
    /// the tensor cores read it through one, scaled or packed, and as it is otherwise.
    pub(crate) fn landed_for<S: Numeric>(
        &self,
        #[comptime] side: Side,
        #[comptime] out: Space,
        #[comptime] instruction: Instruction,
    ) -> Tile {
        let scaled = self.quantized();
        let packing = self.packing();
        if comptime!(instruction == Instruction::Cmma && (scaled || packing != Packing::Plain)) {
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
        #[comptime] out: Space,
    ) -> Tile {
        let space = comptime!(self.place.space.clone());
        let units = self.units();
        let planes = comptime!(plane_windows(&space, &self.place.levels));
        let (stage, mut window) = Memory::landing::<S>(comptime!(space.clone()), units, planes);
        let landing = Tile::new::<S>(stage.kind, comptime!(self.place.clone()));

        // A load is read whole, a tile of several axes included, and landed a run of its finest
        // axis at a time, each run under the scale at its own coordinates.
        let load = self.vector_tile();
        let (sw, (finest, run)) = comptime!((load.values(), load.extents()[0]));
        let size!(SW) = sw;
        let size!(RUN) = run;
        let view = self.nd_packed::<S, SW>(comptime!(Guard::Checked));
        let acc_axes = comptime!(accumulator_axes(side, &out, &space));
        let scales = self.reader(
            comptime!(MatrixAxes::trailing(&space)),
            0usize,
            side,
            comptime!(out.clone()),
            acc_axes,
        );
        let lines = load.count(&space);
        let rank = comptime!(space.rank());
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
            for r in 0..comptime!(sw / run) {
                let offset = comptime!(r * run);
                let mut at = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    at.push(
                        start.at(p) + comptime!(load.offset_along(offset, space.axis_at(p)) as u32),
                    );
                }
                let mut values = Vector::<S, RUN>::empty();
                #[unroll]
                for v in 0..run {
                    values.insert(v, held.extract(offset + v));
                }
                let landed = scales.apply_at::<S, RUN>(values, &at);
                if mine < lines {
                    let base = offset_of(&at, comptime!(strides.clone()));
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
    /// This landing loaded into `frag`, dense over the window's axes.
    pub(crate) fn load_into<E: Numeric>(&self, frag: &mut Matrix<E>, #[comptime] out: Space) {
        let cols = comptime!(landed_cols(&self.place.space, &out) as u32);
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

/// The column count of a landed window's matrix: the trailing axes on the innermost axis's side
/// of the contraction.
fn landed_cols(window: &Space, out: &Space) -> usize {
    let rank = window.rank();
    let innermost = out.contains(window.axis_at(rank - 1));
    (0..rank)
        .rev()
        .take_while(|&p| out.contains(window.axis_at(p)) == innermost)
        .map(|p| window.extent_at(p))
        .product()
}

/// The accumulator's [`MatrixAxes`] read off one factor alone.
fn accumulator_axes(side: Side, out: &Space, own: &Space) -> MatrixAxes {
    let mut col_split = out.rank() - 1;
    let in_columns = |axis: Axis| match side {
        Side::Lhs => !own.contains(axis),
        Side::Rhs => own.contains(axis),
    };
    while col_split > 1 && in_columns(out.axis_at(col_split - 1)) {
        col_split -= 1;
    }
    MatrixAxes {
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

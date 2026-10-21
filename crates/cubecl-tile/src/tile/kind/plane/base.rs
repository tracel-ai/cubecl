//! The plane-level tile ([`PlaneTile`]) and the grid of them one plane owns
//! ([`PlanePartition`]).

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::tile::slices::AxisSlicesKind;
use crate::*;

/// One plane-level tile, by encoding ([`Instruction`]). A warpgroup MMA's is a plane group's
/// rather than one plane's: the group's planes hold it between them, as one plane's units hold a
/// fragment.
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum PlaneTile {
    Cmma(CmmaData),
    Mma(MmaData),
    /// The software leaf's accumulator: a register block, not a hardware fragment.
    Registers(RegisterData),
    /// A plane group's warpgroup MMA accumulator, its four planes' share of one `64 × n` tile.
    Wgmma(WgmmaData),
}

#[cube]
impl PlaneTile {
    /// The semiring this tile contracts under: a hardware instruction runs the sum of products
    /// alone, a register block the one it was opened with.
    pub(crate) fn semiring(&self) -> comptime_type!(Semiring) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => {
                comptime!(Semiring::SUM_PROD)
            }
            PlaneTile::Registers(d) => comptime!(d.accumulation.semiring("PlaneTile::mma")),
        }
    }

    /// Whether this tile contracts through the device's manual MMA.
    pub(crate) fn contracts_through_mma(&self) -> comptime_type!(bool) {
        match self {
            PlaneTile::Mma(_) => comptime!(true),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
        }
    }

    /// Whether this tile contracts through the block-scaled instruction, which reads its factors'
    /// packed codes as they are stored.
    pub(crate) fn reads_codes_as_stored(&self) -> comptime_type!(bool) {
        match self {
            PlaneTile::Mma(data) => comptime!(data.operands == MmaOperands::BlockScaled),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
        }
    }

    /// An uninitialized accumulator tile over the whole `m × n` MMA tile, in `form`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acc<T: Numeric>(
        #[comptime] form: Instruction,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] k: usize,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] accumulation: Accumulation,
    ) -> PlaneTile {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::alloc::<T>(
                MatrixIdent::Accumulator,
                m,
                n,
                k,
                MatrixLayout::RowMajor,
            )),
            Instruction::Mma { io, operands } => PlaneTile::new_Mma(MmaData::acc::<T>(
                m,
                n,
                k,
                MatrixLayout::RowMajor,
                io,
                operands,
            )),
            Instruction::Registers { config } => PlaneTile::new_Registers(
                RegisterData::alloc::<T>(m, n, axes, vector_size, fold, config, accumulation),
            ),
            Instruction::Wgmma => PlaneTile::new_Wgmma(WgmmaData::new::<T>(m, n, axes)),
        }
    }

    /// An uninitialized operand tile in role `ident`, loaded in `layout`, of the form `form`
    /// names. `k` is the operand's own contraction depth, not the instruction's.
    pub(crate) fn operand<T: Numeric>(
        #[comptime] form: Instruction,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> PlaneTile {
        match comptime!(form) {
            Instruction::Cmma => PlaneTile::new_Cmma(CmmaData::alloc::<T>(ident, m, n, k, layout)),
            Instruction::Mma { io, operands } => {
                PlaneTile::new_Mma(MmaData::operand::<T>(ident, m, n, k, layout, io, operands))
            }
            Instruction::Registers { .. } => {
                panic!("PlaneTile::operand: the software form stages no operand plane tile")
            }
            Instruction::Wgmma => panic!(
                "PlaneTile::operand: a warpgroup MMA reads its operands out of shared memory, \
                 into no fragment"
            ),
        }
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the tile's columns: a register
    /// block in place, a cmma fragment bounced through its slot of the plane's scratch on
    /// `sync_plane`.
    pub(crate) fn mul_along<T: Numeric>(&self, factors: &Array<T>, #[comptime] first: usize) {
        match self {
            PlaneTile::Registers(block) => {
                let mut block = block.clone();
                block.mul_along(factors, first);
            }
            PlaneTile::Cmma(fragment) => fragment.mul_along(factors, first),
            PlaneTile::Mma(fragment) => {
                let mut fragment = fragment.clone();
                fragment.mul_along(factors, first);
            }
            PlaneTile::Wgmma(_) => panic!(
                "PlaneTile::mul_along: a warpgroup accumulator is written by the MMAs in flight, \
                 and scaled only once they are done, at its drain"
            ),
        }
    }

    /// This accumulator tile with each cell cast to `E` where its units hold it
    /// ([`MmaData::cast`]). A cmma fragment or a register block is cast as it drains.
    pub(crate) fn cast<T: Numeric, E: Numeric>(&self) -> PlaneTile {
        match self {
            PlaneTile::Mma(fragment) => PlaneTile::new_Mma(fragment.cast::<T, E>()),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => panic!(
                "Tile::cast: a manual-mma accumulator is cast in its registers; a cmma fragment, \
                 a register block or a warpgroup accumulator is cast as it drains"
            ),
        }
    }

    /// Rows of this tile one unit keeps a factor for: a manual fragment's rows the unit holds
    /// cells of, every row of a cmma fragment or a register block.
    pub(crate) fn rows_per_unit<T: Numeric>(&self) -> comptime_type!(usize) {
        match self {
            PlaneTile::Mma(fragment) => fragment.rows_per_unit::<T>(),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) => {
                let (m, _) = self.shape();
                m
            }
            PlaneTile::Wgmma(_) => panic!(
                "PlaneTile::rows_per_unit: a warpgroup accumulator's slices are not read in place; \
                 drain it first"
            ),
        }
    }

    /// This tile carrying its partition's scratch. Only a cmma tile bounces.
    pub(crate) fn with_scratch<T: Numeric>(self, scratch: Shared<[T]>) -> PlaneTile {
        match self {
            PlaneTile::Cmma(d) => PlaneTile::new_Cmma(d.with_scratch(scratch)),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::with_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Spill this tile into its slot of the plane's scratch. Only a cmma tile bounces.
    pub(crate) fn spill_to_scratch<T: Numeric>(&self) {
        match self {
            PlaneTile::Cmma(d) => d.spill_to_scratch::<T>(),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::spill_to_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Add this tile's spilled cells into `mem`.
    pub(crate) fn add_from_scratch<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => d.add_from_scratch::<T, Out>(mem, space),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::add_from_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Commit the warpgroup MMAs this tile issued since its last commit, and return their
    /// completion. Only a warpgroup accumulator contracts asynchronously.
    pub(crate) fn commit<T: Numeric>(&self) -> Pending<()> {
        match self {
            PlaneTile::Wgmma(d) => {
                let mut d = d.clone();
                d.commit::<T>()
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Registers(_) => panic!(
                "Tile::commit: only a plane group's warpgroup accumulator contracts asynchronously"
            ),
        }
    }

    /// The tile's `(m, n)`; only a cmma tile carries it.
    pub(crate) fn shape(&self) -> comptime_type!((usize, usize)) {
        match self {
            PlaneTile::Cmma(d) => comptime!(d.shape),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::shape: only a cmma tile states its shape")
            }
        }
    }

    /// Store this tile, row-major, into `scratch` (one tile's cells).
    pub(crate) fn store_scratch<T: Numeric>(&self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.store_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::store_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch<T: Numeric>(&mut self, scratch: &Shared<[T]>) {
        match self {
            PlaneTile::Cmma(d) => d.load_scratch(scratch),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::load_scratch: only a cmma tile bounces through a scratch")
            }
        }
    }

    pub(crate) fn zero<T: Numeric>(&mut self) {
        match self {
            PlaneTile::Cmma(d) => d.zero::<T>(),
            PlaneTile::Mma(d) => d.zero::<T>(),
            PlaneTile::Registers(d) => d.zero::<T>(),
            PlaneTile::Wgmma(d) => d.zero::<T>(),
        }
    }

    pub(crate) fn init<T: Numeric>(&mut self, val: T) {
        match self {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::init: a hardware mma fragment has no fill other than zero")
            }
            PlaneTile::Registers(d) => d.init::<T>(val),
        }
    }

    pub(crate) fn scale<T: Numeric>(&mut self, factor: T) {
        match self {
            PlaneTile::Cmma(d) => d.scale::<T>(factor),
            PlaneTile::Mma(d) => d.scale::<T>(factor),
            PlaneTile::Registers(d) => d.scale::<T>(factor),
            PlaneTile::Wgmma(_) => panic!(
                "PlaneTile::scale: a warpgroup accumulator is written by the MMAs in flight, and \
                 scaled only once they are done, at its drain"
            ),
        }
    }

    /// Fill this fragment from a memory `src`, a manual-mma one deeper than its instruction as
    /// `load` says.
    pub(crate) fn load_window<T: Numeric>(&mut self, src: &Tile, #[comptime] load: DeepLoad) {
        match self {
            PlaneTile::Cmma(d) => match &src.kind {
                TileKind::Memory(m) => {
                    d.load_window::<T>(m, comptime!(MatrixAxes::edges(&src.place.space).row_split))
                }
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("PlaneTile::load_window: a cmma fragment loads from memory")
                }
            },
            PlaneTile::Mma(d) => d.load_window::<T>(src, load),
            PlaneTile::Registers(_) => {
                panic!("PlaneTile::load_window: a register accumulator is not a fill sink")
            }
            PlaneTile::Wgmma(_) => {
                panic!("PlaneTile::load_window: a warpgroup accumulator is not a fill sink")
            }
        }
    }

    /// Store this tile into `mem`; `space` is the sink window's.
    pub(crate) fn store_window<T: Numeric>(&self, mem: &mut Memory, #[comptime] space: Space) {
        let addressed = mem.store.addressed();
        match self {
            PlaneTile::Cmma(d) => {
                match comptime!(FragmentDrain::of(&mem.access, addressed, true)) {
                    FragmentDrain::Intrinsic => {
                        d.store_window::<T>(mem, comptime!(MatrixAxes::edges(&space).row_split))
                    }
                    FragmentDrain::Bounce => d.bounce_cast_window::<T, T>(mem, space),
                }
            }
            PlaneTile::Mma(d) => d.store_window::<T>(mem, space),
            // Same-type store; the block drains through `store_cast_window`.
            PlaneTile::Registers(d) => d.store_cast_window::<T, T>(mem, space),
            PlaneTile::Wgmma(d) => d.store_cast_window::<T, T>(mem, space),
        }
    }

    /// Add `src`'s block into this one, casting up. Only the register encoding promotes.
    pub(crate) fn add_cast_from<T: Numeric, S: Numeric>(&mut self, src: &PlaneTile) {
        match (self, src) {
            (PlaneTile::Registers(d), PlaneTile::Registers(s)) => d.add_cast_from::<T, S>(s),
            _ => panic!(
                "PlaneTile::add_cast_from: a register block promotes into a register block; a \
                 fragment accumulates in the element its instruction names"
            ),
        }
    }

    /// Store this tile into `mem`, casting; `space` is the sink window's.
    pub(crate) fn store_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        let addressed = mem.store.addressed();
        match self {
            PlaneTile::Cmma(d) => {
                let (m, n) = comptime!(d.shape);
                let in_place = casts_in_place::<T, Out>(m, n);
                match comptime!(FragmentDrain::of(&mem.access, addressed, in_place)) {
                    FragmentDrain::Intrinsic => d.store_cast_window::<T, Out>(
                        mem,
                        comptime!(MatrixAxes::edges(&space).row_split),
                    ),
                    FragmentDrain::Bounce => d.bounce_cast_window::<T, Out>(mem, space),
                }
            }
            PlaneTile::Mma(d) => d.store_cast_window::<T, Out>(mem, space),
            PlaneTile::Registers(d) => d.store_cast_window::<T, Out>(mem, space),
            PlaneTile::Wgmma(d) => d.store_cast_window::<T, Out>(mem, space),
        }
    }
}

/// The `m_tiles × n_tiles` grid of plane tiles one plane owns, row-major: or one plane group, for
/// a warpgroup MMA, whose grid is its one accumulator.
/// `Clone` duplicates the handles, not the tiles.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct PlanePartition {
    pub(crate) frags: Sequence<PlaneTile>,
    #[cube(comptime)]
    pub m_tiles: usize,
    #[cube(comptime)]
    pub n_tiles: usize,
    /// One tile's edges along the window's trailing two axes.
    #[cube(comptime)]
    pub rows: usize,
    #[cube(comptime)]
    pub cols: usize,
    /// This plane's one-tile window of shared memory that a row-wise op bounces through, at the
    /// element the grid holds.
    pub scratch: ComptimeOption<Raw<SharedSlice>>,
    /// How much of this grid the scratch holds.
    #[cube(comptime)]
    pub held: Scratch,
}

#[cube]
impl PlanePartition {
    /// The instruction this grid contracts through where it holds several fragments; `None` where
    /// it holds one.
    pub(crate) fn grid(&self) -> comptime_type!(Option<Instruction>) {
        let several = comptime!(self.m_tiles * self.n_tiles > 1);
        match self.frags.index(0usize) {
            PlaneTile::Cmma(_) => comptime!(several.then_some(Instruction::Cmma)),
            PlaneTile::Mma(d) => comptime!(several.then_some(Instruction::Mma {
                io: d.io,
                operands: d.operands,
            })),
            PlaneTile::Registers(_) => comptime!(None),
            PlaneTile::Wgmma(_) => {
                comptime!(assert!(
                    !several,
                    "PlanePartition: a plane group holds one warpgroup accumulator"
                ));
                comptime!(None)
            }
        }
    }

    /// Whether this grid's tiles are cmma fragments, the one form that drains through a scratch.
    pub(crate) fn is_cmma(&self) -> comptime_type!(bool) {
        match self.frags.index(0usize) {
            PlaneTile::Cmma(_) => comptime!(true),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
        }
    }

    /// The `(mi, ni)` tile (a handle clone); comptime indices only.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> PlaneTile {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The one fragment of a `1 × 1` partition.
    pub(crate) fn fragment(&self) -> PlaneTile {
        comptime!(assert!(
            self.m_tiles == 1 && self.n_tiles == 1,
            "PlanePartition::fragment: {} x {} fragments are stored one at a time; walk the \
             cells and copy each `acc.at(&cell)`",
            self.m_tiles,
            self.n_tiles
        ));
        self.at(0usize, 0usize)
    }

    /// The `sub_m × sub_n` block of fragments `step` selects, or one fragment if `1 × 1`.
    /// A level that cuts the partition must be walked with comptime coordinates.
    pub(crate) fn at_step(&self, step: &Step, #[comptime] space: Space) -> TileKind {
        // A grid of instruction fragments may span a contraction in digits; a register block's
        // edges are its own, and it reads the window's trailing pair.
        let fragments = match self.frags.index(0usize) {
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => comptime!(true),
            PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
        };
        let edges = comptime!(match fragments {
            true => grid_axes(&space, (self.rows, self.cols), (self.m_tiles, self.n_tiles)),
            false => MatrixAxes::edges(&space),
        });
        let a0 = comptime!(space.axis_at(edges.row_split));
        let a1 = comptime!(space.axis_at(edges.col_split));
        // A single-tile axis folds to `0`; a `Dynamic` axis stays runtime (`None`).
        let mi = if comptime!(step.level.tiles_const(&space, a0) == Some(1)) {
            comptime!(Some(0u64))
        } else {
            step.coord(a0).constant()
        };
        let ni = if comptime!(step.level.tiles_const(&space, a1) == Some(1)) {
            comptime!(Some(0u64))
        } else {
            step.coord(a1).constant()
        };
        match comptime!(mi.zip(ni)) {
            Some((c0, c1)) => {
                let (sub_m, sub_n) = comptime!({
                    let (cm, cn) = (step.level.tiles(&space, a0), step.level.tiles(&space, a1));
                    assert!(
                        self.m_tiles.is_multiple_of(cm) && self.n_tiles.is_multiple_of(cn),
                        "Tile::at: the level's grid must divide the partition"
                    );
                    (self.m_tiles / cm, self.n_tiles / cn)
                });
                let mi = comptime!(c0 as usize * sub_m);
                let ni = comptime!(c1 as usize * sub_n);
                if comptime!(sub_m == 1 && sub_n == 1) {
                    TileKind::new_PlaneTile(self.at(mi, ni))
                } else {
                    TileKind::new_PlanePartition(self.window(mi, ni, sub_m, sub_n))
                }
            }
            None => {
                comptime!(assert!(
                    !MatrixGrid::new(&step.level, &space).cuts(),
                    "Tile::at: a level that cuts a partition must be walked with compile-time \
                     coordinates (an unrolled walk)"
                ));
                TileKind::new_PlanePartition(self.clone())
            }
        }
    }

    /// The `m_tiles × n_tiles` sub-partition at `(mi, ni)`, sharing the parent's tiles.
    pub(crate) fn window(
        &self,
        #[comptime] mi: usize,
        #[comptime] ni: usize,
        #[comptime] m_tiles: usize,
        #[comptime] n_tiles: usize,
    ) -> PlanePartition {
        let mut frags = Sequence::<PlaneTile>::new();
        #[unroll]
        for i in 0..m_tiles {
            #[unroll]
            for j in 0..n_tiles {
                frags.push(self.at(comptime!(mi + i), comptime!(ni + j)));
            }
        }
        PlanePartition {
            frags,
            m_tiles,
            n_tiles,
            rows: comptime!(self.rows),
            cols: comptime!(self.cols),
            scratch: self.scratch.clone(),
            held: comptime!(self.held),
        }
    }

    /// This grid with each cell cast to `E`, fragment for fragment ([`PlaneTile::cast`]).
    pub(crate) fn cast<T: Numeric, E: Numeric>(&self) -> PlanePartition {
        let mut frags = Sequence::<PlaneTile>::new();
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            frags.push(self.frags.index(i).cast::<T, E>());
        }
        PlanePartition {
            frags,
            m_tiles: comptime!(self.m_tiles),
            n_tiles: comptime!(self.n_tiles),
            rows: comptime!(self.rows),
            cols: comptime!(self.cols),
            scratch: ComptimeOption::new_None(),
            held: Scratch::None,
        }
    }

    /// `self[r, :] *= corr[r]`, each tile bounced through the scratch.
    ///
    /// The scratch is the plane's own slot (`with_scratch` indexes it by `PLANE_POS`), so
    /// the bounce syncs the plane rather than the cube: every unit of the plane must call it,
    /// and the other planes need not.
    pub(crate) fn rescale_rows<T: Numeric>(&self, corr: &Array<T>) {
        let mut scratch = #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.at::<T>().clone(),
            ComptimeOption::None => panic!(
                "PlanePartition::rescale_rows: a row-wise op on a plane-resident accumulator \
                 bounces through a scratch; open the accumulator with `with_scratch`"
            ),
        };
        let (m, n) = self.at(0usize, 0usize).shape();
        let rows = comptime!(self.m_tiles * m);
        let mut moved = false;
        #[unroll]
        for r in 0..rows {
            if corr[r] != T::from_int(1) {
                moved = true;
            }
        }
        let cells = comptime!(m * n);
        let units = PLANE_DIM as usize;
        let plane_unit = UNIT_POS_PLANE as usize;
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut tile = self.at(mi, ni);
                if moved {
                    tile.store_scratch(&scratch);
                }
                sync_plane();
                if moved {
                    let mut cell = plane_unit;
                    while cell < cells {
                        scratch[cell] *= corr[mi * m + cell / n];
                        cell += units;
                    }
                }
                sync_plane();
                if moved {
                    tile.load_scratch(&scratch);
                }
                sync_plane();
            }
        }
    }

    /// `self[i, :] *= factors[i]` for every slice `i` along the grid's columns, counted down the
    /// whole grid. Each tile is scaled where it is held ([`PlaneTile::mul_along`]), unless the
    /// scratch holds every fragment of the grid at once: then the plane spills them all, meets
    /// once, scales, meets once, and reloads them all.
    pub(crate) fn mul_along<T: Numeric>(&self, factors: &Array<T>) {
        // Only a cmma grid is opened with a scratch, so one that holds the whole grid is one.
        if comptime!(self.held == Scratch::WholeGrid) {
            self.mul_along_whole_grid(factors);
        } else {
            #[unroll]
            for mi in 0..self.m_tiles {
                #[unroll]
                for ni in 0..self.n_tiles {
                    let tile = self.at(mi, ni);
                    let rows = tile.rows_per_unit::<T>();
                    tile.mul_along(factors, comptime!(mi * rows));
                }
            }
        }
    }

    /// [`mul_along`](Self::mul_along) over a grid of cmma fragments each with a slot of its own.
    fn mul_along_whole_grid<T: Numeric>(&self, factors: &Array<T>) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            self.cmma_at(i).spill_to_scratch::<T>();
        }
        sync_plane();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                self.cmma_at(comptime!(mi * self.n_tiles + ni))
                    .mul_spilled(factors, comptime!(mi * self.rows));
            }
        }
        sync_plane();
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            self.cmma_at(i).reload_from_scratch::<T>();
        }
        sync_plane();
    }

    /// The `i`-th tile of a grid of cmma fragments, in row-major order.
    fn cmma_at(&self, #[comptime] i: usize) -> CmmaData {
        match self.frags.index(i).clone() {
            PlaneTile::Cmma(fragment) => fragment,
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                panic!("PlanePartition: a grid holds one kind of tile")
            }
        }
    }

    /// An uninitialized plane-resident accumulator partition mirroring `space`'s grid.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mirror<T: Numeric>(
        #[comptime] space: Space,
        #[comptime] axes: MatrixAxes,
        #[comptime] form: Instruction,
        #[comptime] grid: GridShape,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] accumulation: Accumulation,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile {
        let (m_tiles, n_tiles) = comptime!(grid.tiles);
        let (m, n, k) = comptime!((grid.m, grid.n, grid.k));

        let mut frags = Sequence::<PlaneTile>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                frags.push(PlaneTile::acc::<T>(
                    form,
                    m,
                    n,
                    axes,
                    k,
                    vector_size,
                    fold,
                    accumulation,
                ));
            }
        }
        Tile {
            kind: TileKind::new_PlanePartition(PlanePartition {
                frags,
                m_tiles,
                n_tiles,
                rows: m,
                cols: n,
                scratch: ComptimeOption::new_None(),
                held: Scratch::None,
            }),
            // A `Dynamic` extent here is never read; fragments were sized from the statement.
            factor: Factor::none(),
            place: comptime!(Placement::new(space, depth, levels)),
            element: elem_type_of::<T>(),
        }
    }

    /// Uninitialized operand fragments for one region under `out`'s contraction, of the form
    /// `form` names. `m`/`n` are the accumulator fragment's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn store<T: Numeric>(
        #[comptime] window: Space,
        #[comptime] form: Instruction,
        #[comptime] out: Space,
        #[comptime] grid: (usize, usize),
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] depth: usize,
        #[comptime] levels: Vec<Level>,
    ) -> Tile {
        let edges = comptime!(MatrixAxes::operand(&window, &out));
        let a0 = comptime!(window.axis_at(edges.row_split));
        let a1 = comptime!(window.axis_at(edges.col_split));

        // The contracted axis is the one `out` lacks; `A` shares its rows, `B` its columns.
        let (contracted, free) = comptime!(match (out.contains(a0), out.contains(a1)) {
            (false, true) => (a0, a1),
            (true, false) => (a1, a0),
            _ => panic!(
                "PlanePartition::store: one of the window's matrix axes is contracted and the \
                 other is the accumulator's"
            ),
        });
        let out_edges = comptime!(MatrixAxes::edges(&out));
        let out_rows = comptime!(out.axis_at(out_edges.row_split));
        let (ident, tiles) = comptime!(if free == out_rows {
            (MatrixIdent::A, grid.0)
        } else {
            assert!(
                free == out.axis_at(out_edges.col_split),
                "PlanePartition::store: the operand's free axis must be one of the output's \
                 trailing two"
            );
            (MatrixIdent::B, grid.1)
        });
        let (t0, t1) = comptime!(if free == a0 { (tiles, 1) } else { (1, tiles) });
        // A fragment's edge along the free axis as the window holds it: the instruction's, or
        // fewer where the product's rows end inside one fragment and the rest read as zeros.
        let rows_along_free = comptime!(
            (if ident == MatrixIdent::A { m } else { n }).min(window.extent(free) / tiles)
        );
        // Every digit of the contraction: the edge of the group `contracted` leads.
        let k = comptime!(if contracted == a0 {
            edges.rows(&window)
        } else {
            edges.cols(&window)
        });
        let (e0, e1) = comptime!(if free == a0 {
            (rows_along_free, k)
        } else {
            (k, rows_along_free)
        });
        // A window listing the role's axes reversed (a weight `{n, k}`) loads col-major.
        let layout = comptime!(if (ident == MatrixIdent::A) == (contracted == a1) {
            MatrixLayout::RowMajor
        } else {
            MatrixLayout::ColMajor
        });

        let mut frags = Sequence::<PlaneTile>::new();
        #[unroll]
        for _i in 0..t0 {
            #[unroll]
            for _j in 0..t1 {
                frags.push(PlaneTile::operand::<T>(form, ident, m, n, k, layout));
            }
        }
        Tile {
            kind: TileKind::new_PlanePartition(PlanePartition {
                frags,
                m_tiles: t0,
                n_tiles: t1,
                rows: e0,
                cols: e1,
                scratch: ComptimeOption::new_None(),
                held: Scratch::None,
            }),
            factor: Factor::none(),
            place: comptime!(Placement::new(window, depth, levels)),
            element: elem_type_of::<T>(),
        }
    }

    /// This region of an operand, in the form `instruction` reads it: as it is where it already
    /// holds fragments or the form is a register block, else loaded into fragments contracting
    /// into `acc`, manual-mma ones deeper than their instruction as `load` says.
    pub(crate) fn operand<T: Numeric>(
        src: &Tile,
        acc: &Tile,
        #[comptime] instruction: Instruction,
        #[comptime] load: DeepLoad,
    ) -> Tile {
        match &src.kind {
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => src.clone(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => match comptime!(instruction) {
                // A warpgroup MMA reads the window itself, through a descriptor.
                Instruction::Registers { .. } | Instruction::Wgmma => src.clone(),
                Instruction::Cmma | Instruction::Mma { .. } => {
                    PlanePartition::fragments_in::<T>(src, acc, instruction, load)
                }
            },
        }
    }

    fn fragments_in<T: Numeric>(
        src: &Tile,
        acc: &Tile,
        #[comptime] form: Instruction,
        #[comptime] load: DeepLoad,
    ) -> Tile {
        let gathered = src.gathered();
        comptime!(assert!(
            !gathered,
            "PlanePartition::fragments: a gathered operand cannot load into fragments; stage it \
             into shared memory first"
        ));
        let (grid, m, n) = acc.fragment_grid();
        let scaled = src.quantized();
        let packing = src.packing();
        let shared = src.is_shared();
        // The operand's own depth: the extent of the axes the accumulator lacks.
        let k = comptime!(
            src.place
                .space
                .axes()
                .filter(|&axis| !acc.place.space.contains(axis))
                .map(|axis| src.place.space.extent(axis))
                .product::<usize>()
        );
        let block_scaled = comptime!(matches!(
            form,
            Instruction::Mma {
                operands: MmaOperands::BlockScaled,
                ..
            }
        ));
        if comptime!(block_scaled) {
            assert_block_scaled::<T>(src, m, n, k);
        }
        let mut frags = PlanePartition::store::<T>(
            comptime!(src.place.space.clone()),
            comptime!(form),
            comptime!(acc.place.space.clone()),
            comptime!(grid),
            comptime!(m),
            comptime!(n),
            comptime!(src.place.depth),
            comptime!(src.place.levels.clone()),
        );
        let lane_runs = match comptime!(form) {
            Instruction::Mma { .. } => lane_run_steps::<T>(m, n, k),
            Instruction::Cmma | Instruction::Registers { .. } | Instruction::Wgmma => {
                comptime!(1usize)
            }
        };
        if comptime!(block_scaled || lane_runs > 1) {
            // The instruction reads the stored values and their scales as they lie, or a
            // fragment deeper than it decodes them as it loads: nothing lands on the way, so each
            // fragment loads its own window of the source.
            match &frags.kind {
                TileKind::PlanePartition(partition) => partition.fill_from::<T>(src, load),
                TileKind::Memory(_)
                | TileKind::PlaneTile(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("PlanePartition::fragments: operand fragments are a partition")
                }
            }
        } else if comptime!(scaled || packing != Packing::Plain || !shared) {
            // A scaled, packed or global operand loads from its plane's landing, as the direct
            // contraction does: a fragment reads a shared, plain window as it lies.
            let side = comptime!(Side::of(&src.place.space, &acc.place.space));
            let landing = src.landed::<T>(side, comptime!(acc.place.space.clone()));
            frags.copy_from(&landing);
            // The landing is this region's until every unit's load has read it.
            sync_plane();
        } else {
            frags.copy_from(src);
        }
        frags
    }

    /// Fill each tile from its window of `src`, in row-major order, manual-mma tiles deeper than
    /// their instruction as `load` says.
    pub(crate) fn fill_from<T: Numeric>(&self, src: &Tile, #[comptime] load: DeepLoad) {
        let level = comptime!(fragment_level(
            &src.place.space,
            (self.rows, self.cols),
            (self.m_tiles, self.n_tiles)
        ));
        let edges = comptime!(grid_axes(
            &src.place.space,
            (self.rows, self.cols),
            (self.m_tiles, self.n_tiles)
        ));
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.at(&Region::cell(
                    comptime!(src.place.depth),
                    comptime!(src.place.space.clone()),
                    comptime!(level.clone()),
                    edges,
                    mi,
                    ni,
                ));
                frag.load_window::<T>(&window, load);
            }
        }
    }

    /// Zero every tile.
    pub(crate) fn zero<T: Numeric>(&self) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.zero::<T>();
        }
    }

    /// Initialize every tile with `val`.
    pub(crate) fn init<T: Numeric>(&self, val: T) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.init::<T>(val);
        }
    }

    /// Multiply every tile by `factor`.
    pub(crate) fn scale<T: Numeric>(&self, factor: T) {
        #[unroll]
        for i in 0..comptime!(self.m_tiles * self.n_tiles) {
            let mut frag = self.frags.index(i).clone();
            frag.scale::<T>(factor);
        }
    }
}

#[cube]
impl PlanePartition {
    /// This grid's slices along its columns: a unit's register block where the grid is one, the
    /// fragments otherwise.
    pub(crate) fn slices(&self) -> AxisSlicesKind {
        match self.at(0usize, 0usize) {
            PlaneTile::Registers(block) => {
                comptime!(assert!(
                    self.m_tiles * self.n_tiles == 1,
                    "Tile::along: a unit's slices are one register block"
                ));
                AxisSlicesKind::new_Registers(block)
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => AxisSlicesKind::new_Fragments(self.clone()),
            PlaneTile::Wgmma(_) => panic!(
                "Tile::along: a warpgroup accumulator's slices are not read in place; drain it first"
            ),
        }
    }

    /// `self[r, c] = self[r, c] · scale + bias[r, c]` over every fragment, `bias` read at each
    /// cell's coordinates in the window over `space`.
    pub(crate) fn scale_add_along<E: Float>(
        &self,
        scale: E,
        bias: &Procedural,
        #[comptime] space: Space,
    ) {
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                let mut fragment = self.at(mi, ni).manual_fragment();
                fragment.scale_add_along::<E>(
                    scale,
                    bias,
                    comptime!(space.clone()),
                    comptime!((mi * self.rows, ni * self.cols)),
                );
            }
        }
    }

    /// Each row's max, starting from `seed`'s: every fragment's cells on it, then the units'.
    pub(crate) fn maxima_along<E: Float>(&self, seed: &Array<E>) -> Array<E> {
        let rows = self.at(0usize, 0usize).rows_per_unit::<E>();
        let mut maxima = Array::<E>::new(comptime!(self.m_tiles * rows));
        #[unroll]
        for i in 0..comptime!(self.m_tiles * rows) {
            maxima[i] = seed[i];
        }
        self.reduce_along(&mut maxima, Monoid::Max);
        maxima
    }

    /// Each row's sum: every fragment's cells on it, then the units'.
    pub(crate) fn sums_along<E: Float>(&self) -> Array<E> {
        let rows = self.at(0usize, 0usize).rows_per_unit::<E>();
        let mut sums = Array::<E>::new(comptime!(self.m_tiles * rows));
        #[unroll]
        for i in 0..comptime!(self.m_tiles * rows) {
            sums[i] = E::from_int(0);
        }
        self.reduce_along(&mut sums, Monoid::Sum);
        sums
    }

    /// `values`, a slot for every row a unit holds cells of, written at the rows of `window`, one
    /// cell a row, each row by one unit ([`MmaData::store_along`]).
    pub(crate) fn store_along<E: Float>(&self, values: &Array<E>, window: &mut Memory) {
        let rows = self.at(0usize, 0usize).rows_per_unit::<E>();
        #[unroll]
        for mi in 0..self.m_tiles {
            self.at(mi, 0usize).manual_fragment().store_along::<E>(
                values,
                window,
                comptime!(mi * rows),
                comptime!(mi * self.rows),
            );
        }
    }

    /// `self[r, c] = exp(self[r, c] − slices[r])` over every fragment.
    pub(crate) fn exp_minus_along<E: Float>(&self, slices: &Array<E>) {
        let rows = self.at(0usize, 0usize).rows_per_unit::<E>();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                let mut fragment = self.at(mi, ni).manual_fragment();
                fragment.exp_minus_along(slices, comptime!(mi * rows));
            }
        }
    }

    /// `slots` combined under `monoid` with every fragment's cells on each row, the unit's own
    /// first and then across the units that share the row: one exchange a row, however many
    /// fragments it spans.
    fn reduce_along<E: Float>(&self, slots: &mut Array<E>, #[comptime] monoid: Monoid) {
        let rows = self.at(0usize, 0usize).rows_per_unit::<E>();
        #[unroll]
        for mi in 0..self.m_tiles {
            #[unroll]
            for ni in 0..self.n_tiles {
                self.at(mi, ni).manual_fragment().reduce_own_cells(
                    slots,
                    comptime!(mi * rows),
                    monoid,
                );
            }
            self.at(mi, 0usize).manual_fragment().reduce_across_units(
                slots,
                comptime!(mi * rows),
                monoid,
            );
        }
    }
}

#[cube]
impl PlaneTile {
    /// This tile's slices along its columns: a register block's, or one fragment's.
    pub(crate) fn slices(&self) -> AxisSlicesKind {
        match self {
            PlaneTile::Registers(block) => AxisSlicesKind::new_Registers(block.clone()),
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => AxisSlicesKind::new_Fragment(self.clone()),
            PlaneTile::Wgmma(_) => panic!(
                "Tile::along: a warpgroup accumulator's slices are not read in place; drain it first"
            ),
        }
    }

    /// This tile's manual-mma fragment, whose cells are read where its units hold them. A cmma
    /// fragment's cells lie where the instruction keeps them, unknown to the program, and are
    /// read after draining it into its plane's window.
    pub(crate) fn manual_fragment(&self) -> MmaData {
        match self {
            PlaneTile::Mma(fragment) => fragment.clone(),
            PlaneTile::Cmma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => panic!(
                "PlaneTile: a cmma fragment's slices are read after draining it into its plane's \
                 window; a manual-mma fragment's are read in place"
            ),
        }
    }
}

/// The matrix a grid of `tiles` fragments of `frag` each spans in `window`: the window's
/// trailing pair where its edges are the grid's, else its axes grouped to the grid's two edges,
/// so an edge spelled by several axes (a contraction in digits) is one edge.
fn grid_axes(window: &Space, frag: (usize, usize), tiles: (usize, usize)) -> MatrixAxes {
    let (rows, cols) = (frag.0 * tiles.0, frag.1 * tiles.1);
    let trailing = MatrixAxes::edges(window);
    if (trailing.rows(window), trailing.cols(window)) == (rows, cols) {
        return trailing;
    }
    let mut grouped = MatrixAxes::new(window, rows, cols).unwrap_or(trailing);
    // Axes of one leading the row group are batch: the group starts at the axis a grid cuts.
    while grouped.row_split + 1 < grouped.col_split && window.extent_at(grouped.row_split) == 1 {
        grouped.row_split += 1;
    }
    grouped
}

/// The never-walked level cutting an operand's window into the partition's fragments: the
/// window's matrix grouped to `frag` times `tiles` along each edge, an edge spelled by several
/// axes (a contraction in digits) held whole by every fragment, an edge of one axis cut.
fn fragment_level(window: &Space, frag: (usize, usize), tiles: (usize, usize)) -> Level {
    let edges = grid_axes(window, frag, tiles);
    let (p0, p1, rank) = (edges.row_split, edges.col_split, window.rank());
    // The one axis longer than one an edge's group spans, where it spans one: the axis a grid
    // of fragments cuts along that edge.
    let only = |group: core::ops::Range<usize>| {
        let mut long = group.filter(|&p| window.extent_at(p) > 1);
        match (long.next(), long.next()) {
            (Some(p), None) => Some(p),
            _ => None,
        }
    };
    let (rows_axis, cols_axis) = (only(p0..p1), only(p1..rank));
    // The fragment edge and count at position `p`, where it is the one axis of its edge.
    let cut = |p: usize| match Some(p) {
        q if q == rows_axis => Some((frag.0, tiles.0)),
        q if q == cols_axis => Some((frag.1, tiles.1)),
        _ => None,
    };
    let axes: Vec<Axis> = window.axes().collect();
    let leaf: Vec<(Axis, usize)> = axes
        .iter()
        .enumerate()
        .map(|(p, &axis)| (axis, cut(p).map_or(window.extent(axis), |(edge, _)| edge)))
        .collect();
    let counts: Vec<(Axis, usize)> = axes
        .iter()
        .enumerate()
        .map(|(p, &axis)| (axis, cut(p).map_or(1, |(_, count)| count)))
        .collect();
    for (&(axis, edge), &(_, count)) in leaf.iter().zip(&counts) {
        assert!(
            edge * count == window.extent(axis),
            "PlanePartition::fill_from: {count} fragments of {edge} do not cover the {} of {axis:?}",
            window.extent(axis)
        );
    }
    assert!(
        (rows_axis.is_some() || tiles.0 == 1) && (cols_axis.is_some() || tiles.1 == 1),
        "PlanePartition::fill_from: an edge spelled by several axes is a contraction every \
         fragment holds whole, and {tiles:?} fragments of {frag:?} cut one of {window:?}"
    );
    Levels::leaf(&leaf).walk(&counts).build().remove(0)
}

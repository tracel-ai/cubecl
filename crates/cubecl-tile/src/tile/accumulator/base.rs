//! The plane-resident accumulator an output opens ([`Tile::accumulator`]).

use cubecl::prelude::*;

use crate::ops::matmul::leaf::memory;
use crate::*;

/// The shape a plane-resident accumulator is opened at: `tiles` per plane, each `m × n`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GridShape {
    pub(crate) tiles: (usize, usize),
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl GridShape {
    /// The grid an accumulator at `place` contracting `lhs` holds.
    pub(crate) fn new(place: &Placement, lhs: &Space) -> Self {
        let levels = place.below();
        let tiles = partition_shape(&place.space, levels);
        let leaf = place.space.leaf(levels);
        let lhs_leaf = lhs.leaf(levels);
        let axes = MatrixAxes::accumulator(&leaf, &lhs_leaf);
        // Use the accumulator's own edges: a split column group is one edge.
        GridShape {
            tiles,
            m: axes.rows(&leaf),
            n: axes.cols(&leaf),
            k: lhs_leaf.contracted_extent(&place.space),
        }
    }
}

/// How much of a plane's accumulator its scratch holds at once. Serialized names are persisted in
/// autotune keys and must not change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Scratch {
    /// No window: fragments store through their intrinsic, which cannot add; a drain that needs
    /// one opens [`OneTile`](Scratch::OneTile).
    None,
    /// One tile, bounced through shared memory with three cube-wide barriers each.
    OneTile,
    /// The plane's whole grid, two barriers for the whole drain.
    #[serde(rename = "WholePartition")]
    WholeGrid,
}

/// The persisted setting's spelling.
impl core::fmt::Display for Scratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Scratch::None => write!(f, "none"),
            Scratch::OneTile => write!(f, "one_tile"),
            Scratch::WholeGrid => write!(f, "whole_partition"),
        }
    }
}

impl Scratch {
    /// Slots the window holds for a grid of `tiles`.
    pub fn slots(self, tiles: usize) -> usize {
        match self {
            Scratch::None => 0,
            Scratch::OneTile => 1,
            Scratch::WholeGrid => tiles,
        }
    }

    /// The slot tile `i` of a grid of `tiles` spills into.
    pub(crate) fn slot_of(self, i: usize, tiles: usize) -> usize {
        match self {
            Scratch::WholeGrid => i,
            Scratch::None | Scratch::OneTile => {
                let _ = (i, tiles);
                0
            }
        }
    }

    /// Whether a fragment goes through shared memory on its way out; forced when folding.
    pub fn bounces(self) -> bool {
        !matches!(self, Scratch::None)
    }

    /// This setting raised to what `folds` demands.
    pub fn at_least_bouncing(self, folds: bool) -> Scratch {
        match (folds, self) {
            (true, Scratch::None) => Scratch::OneTile,
            (_, scratch) => scratch,
        }
    }
}

/// The planes `levels` distribute `space` across, a plane group's every plane among them: one
/// window each, as `PLANE_POS` indexes them. Panics on a runtime-only count.
///
/// A plane level spanning the launch deals its tiles to the launch's planes, not to the cube's,
/// so the cube holds the planes its order groups in one, whatever the tiles.
pub(crate) fn plane_windows(space: &Space, levels: &[Level]) -> usize {
    let mut handed = space.clone();
    let mut planes = 1;
    for level in levels {
        if let Some(order) = level.plane_order() {
            planes *= order.planes_per_cube();
        } else if let Some(cube) = level.unit_cube() {
            planes *= cube.planes;
        } else if let Some(per_instance) = level.coverage().scope().and_then(ComputeScope::planes) {
            let mut instances = 1;
            for axis in level.axes() {
                if level.distributes(axis) {
                    instances *= level.instances_along(&handed, axis).unwrap_or_else(|| {
                        panic!(
                            "Tile::with_scratch: {axis:?} is distributed across the cube's planes \
                             at a count only the launch knows, so the plane's windows cannot be \
                             sized"
                        )
                    });
                }
            }
            planes *= instances * per_instance;
        }
        handed = level.child(&handed);
    }
    planes
}

/// What an output does with the sum contracted into it: open, scratch and drain an accumulator.
#[cube]
impl Tile {
    /// What one plane sums `lhs · rhs` into under `semiring`, in the form `instruction` names,
    /// at the element `EA`, opened at the semiring's identity.
    pub fn accumulator<EA: Numeric>(
        &self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] instruction: Instruction,
        #[comptime] semiring: Semiring,
    ) -> Tile {
        match comptime!(instruction) {
            Instruction::Registers { config } => {
                register_accumulator::<EA>(self, lhs, rhs, config, semiring)
            }
            Instruction::Cmma | Instruction::Mma { .. } | Instruction::Wgmma => {
                if comptime!(instruction == Instruction::Wgmma) {
                    comptime!(assert_warpgroups_hold(&self.place));
                    comptime!(assert_groups_own_their_box(&self.place));
                }
                let vector_size = self.vector_size();
                accumulator_in::<EA>(
                    self,
                    lhs,
                    instruction,
                    vector_size,
                    1usize,
                    comptime!(Accumulation::Contraction(semiring)),
                )
            }
        }
    }

    /// A register block over this tile's grid for a reduction over `input` under `monoid`, at
    /// the element `EA`, opened at its identity.
    pub fn block_reducer<EA: Numeric>(
        &self,
        input: &Tile,
        #[comptime] config: RegisterBlock,
        #[comptime] monoid: Monoid,
    ) -> Tile {
        let vector_size = self.vector_size();
        accumulator_in::<EA>(
            self,
            input,
            comptime!(Instruction::Registers { config }),
            vector_size,
            1usize,
            comptime!(Accumulation::Reduction(monoid)),
        )
    }

    /// This accumulator opened with a per-plane shared-memory scratch for bouncing drains, or
    /// as it is for [`Scratch::None`]. [`drained_into`](Tile::drained_into) opens the smallest
    /// one itself where it needs one; stating it chooses how much of the grid it holds, or serves
    /// a drain region by region.
    pub fn with_scratch(self, #[comptime] scratch: Scratch) -> Tile {
        let define!(Served) = self.element;
        match comptime!(scratch) {
            Scratch::None => self,
            _ => match &self.kind {
                TileKind::PlanePartition(p) => {
                    let (m, n) = p.at(0usize, 0usize).shape();
                    let cells = comptime!(m * n);
                    let tiles = comptime!(p.m_tiles * p.n_tiles);
                    let slots = comptime!(scratch.slots(tiles));
                    // Indexed by hardware plane position, not level position: the launch decides the cube's shape.
                    let planes = comptime!(plane_windows(&self.place.space, &self.place.levels));
                    let plane = PLANE_POS.cast::<usize>() * comptime!(cells * slots);
                    let shared = Shared::<[Served]>::new_slice(comptime!(cells * slots * planes));
                    // Each fragment carries its own slot, so drain order doesn't matter.
                    let mut frags = Sequence::<PlaneTile>::new();
                    #[unroll]
                    for i in 0..tiles {
                        let start = plane + comptime!(cells * scratch.slot_of(i, tiles));
                        let slot = shared.clone().map(|s| &s[start..start + cells]);
                        frags.push(p.frags.index(i).clone().with_scratch(slot));
                    }
                    Tile {
                        kind: TileKind::new_PlanePartition(PlanePartition {
                            frags,
                            m_tiles: comptime!(p.m_tiles),
                            n_tiles: comptime!(p.n_tiles),
                            rows: comptime!(p.rows),
                            cols: comptime!(p.cols),
                            scratch: ComptimeOption::new_Some(Raw::<SharedSlice>::of::<Served>(
                                shared.clone().map(|s| &s[plane..plane + cells]),
                            )),
                            held: comptime!(scratch),
                        }),
                        factor: self.factor.clone(),
                        place: comptime!(self.place.clone()),
                        element: comptime!(self.element),
                    }
                }
                TileKind::Memory(_)
                | TileKind::PlaneTile(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("Tile::with_scratch: a scratch backs a plane-resident accumulator")
                }
            },
        }
    }

    /// Drain this accumulator into `dest`, cast to its element. `self` and `dest` must be
    /// indexed by the same region. A cmma grid opened with no scratch opens one where `dest`
    /// needs its fragments bounced.
    pub fn drained_into(&self, dest: &Tile) {
        let define!(Served) = self.element;
        let define!(Out) = dest.element;
        // A sum written out is a sum with its scales applied: multiplied by them once, as it
        // drains. A grid of cmma fragments is scaled, and reaches a destination it cannot store
        // to directly, through a scratch: where none was opened, the drain opens the smallest.
        let levels = self.factor_levels();
        let scaled = comptime!(levels > 0);
        let unopened = self.scratch_unopened();
        let in_place = self.casts_in_place_to::<Served, Out>();
        let bounces = dest.fragments_bounce(in_place);
        if comptime!(unopened && (scaled || bounces)) {
            let opened = self.clone().with_scratch(Scratch::OneTile);
            opened.drained_into(dest);
        } else if comptime!(scaled) {
            let mut sum = self.unscaled();
            sum.scale_by::<f32>(self.factor.whole());
            sum.drained_into(dest);
        } else {
            self.drain_as_opened::<Served, Out>(dest);
        }
    }
}

#[cube]
impl Tile {
    /// This plane-resident accumulator with each cell cast to `E` where its units hold it: what
    /// an attention contracts its probabilities with the values through, the next contraction
    /// reading the cast accumulator in its registers, without a round trip through shared memory.
    ///
    /// Only a manual-mma accumulator is cast in place; any other is cast as it drains.
    pub fn cast<E: Numeric>(&self) -> Tile {
        let define!(Served) = self.element;
        let kind = match &self.kind {
            TileKind::PlanePartition(partition) => {
                TileKind::new_PlanePartition(partition.cast::<Served, E>())
            }
            TileKind::PlaneTile(tile) => TileKind::new_PlaneTile(tile.cast::<Served, E>()),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::cast: only a plane-resident accumulator is cast in its registers")
            }
        };
        Tile {
            kind,
            factor: self.factor.clone(),
            place: comptime!(self.place.clone()),
            element: elem_type_of::<E>(),
        }
    }

    /// Whether this accumulator's fragments cast to `Out` are fragments the device holds
    /// ([`casts_in_place`]): what decides whether a cmma grid draining into an `Out` destination
    /// stores whole or bounces.
    fn casts_in_place_to<A: Numeric, Out: Numeric>(&self) -> comptime_type!(bool) {
        // Only a cmma fragment has a cast the device may lack. Reading the grid of anything else
        // would ask the extent of an axis that may be dynamic.
        match &self.kind {
            // The fragment's own edges: the placement's space spans every plane's grid.
            TileKind::PlanePartition(p) => {
                let cmma = p.is_cmma();
                if comptime!(cmma) {
                    casts_in_place::<A, Out>(comptime!(p.rows), comptime!(p.cols))
                } else {
                    comptime!(true)
                }
            }
            TileKind::PlaneTile(t) => match t {
                PlaneTile::Cmma(_) => {
                    let (_, m, n) = self.fragment_grid();
                    casts_in_place::<A, Out>(m, n)
                }
                PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                    comptime!(true)
                }
            },
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(true),
        }
    }

    /// Waits for the units that share the scratch this accumulator bounces through: its holder's.
    /// A plane's windows are its own ([`plane_windows`]), so its plane alone meets them; a plane
    /// group's fragments span its planes, so the cube does.
    fn sync_scratch(&self) {
        match comptime!(self.place.holder()) {
            ComputeScope::Unit | ComputeScope::Plane => sync_plane(),
            ComputeScope::PlaneGroup { .. } | ComputeScope::Cube => sync_cube(),
        }
    }

    /// Whether this is a grid of cmma fragments opened with no scratch.
    fn scratch_unopened(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::PlanePartition(p) => {
                let cmma = p.is_cmma();
                comptime!(cmma && p.held == Scratch::None)
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// [`drained_into`](Tile::drained_into) for the fragments of chunk `turn` alone, of
    /// `chunks` the fragment grid is dealt into by place: what one writer of a cyclic schedule
    /// lands in one round, each fragment drained in exactly one of the `chunks` rounds. It reaches
    /// every barrier a whole drain does, so writers landing different chunks meet at all of them.
    /// A grid opened with no scratch drains through the smallest, as a whole drain does.
    pub(crate) fn drained_chunk_into<A: Numeric, Out: Numeric>(
        &self,
        dest: &Tile,
        turn: usize,
        #[comptime] chunks: usize,
    ) {
        let unopened = self.scratch_unopened();
        let in_place = self.casts_in_place_to::<A, Out>();
        let bounces = dest.fragments_bounce(in_place);
        if comptime!(unopened && bounces) {
            let opened = self.clone().with_scratch(Scratch::OneTile);
            opened.drained_chunk_into::<A, Out>(dest, turn, chunks);
        } else {
            self.drain_chunk_as_opened::<A, Out>(dest, turn, chunks);
        }
    }

    /// [`drained_chunk_into`](Tile::drained_chunk_into) through the scratch this grid was opened
    /// with, where `dest` is one its fragments must bounce to reach.
    fn drain_chunk_as_opened<A: Numeric, Out: Numeric>(
        &self,
        dest: &Tile,
        turn: usize,
        #[comptime] chunks: usize,
    ) {
        let in_place = self.casts_in_place_to::<A, Out>();
        let bounces = dest.fragments_bounce(in_place);
        match &self.kind {
            TileKind::PlanePartition(p) => match comptime!(DrainPlan::new(p.held, bounces)) {
                DrainPlan::Straight => drain_chunk_below::<A, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Copy),
                    0usize,
                    turn,
                    chunks,
                ),
                DrainPlan::BounceEach => drain_chunk_below::<A, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Bounce),
                    0usize,
                    turn,
                    chunks,
                ),
                DrainPlan::BounceTogether => {
                    self.sync_scratch();
                    drain_chunk_below::<A, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Spill),
                        0usize,
                        turn,
                        chunks,
                    );
                    self.sync_scratch();
                    drain_chunk_below::<A, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Add),
                        0usize,
                        turn,
                        chunks,
                    );
                    self.sync_scratch();
                }
            },
            // One tile: it is the one chunk of its grid. A fragment bouncing into its destination
            // meets its holder at its barriers whatever the turn, so it bounces through the
            // predicated pass; anything else stores with no barrier at all.
            TileKind::PlaneTile(t) => {
                let cmma = match t {
                    PlaneTile::Cmma(_) => comptime!(true),
                    PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                        comptime!(false)
                    }
                };
                let in_place = self.casts_in_place_to::<A, Out>();
                let bounces = dest.fragments_bounce(in_place);
                if comptime!(cmma && bounces) {
                    drain_chunk_leaf::<A, Out>(self, dest, comptime!(DrainPass::Bounce), turn == 0);
                } else {
                    drain_chunk_leaf::<A, Out>(self, dest, comptime!(DrainPass::Copy), turn == 0);
                }
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::drained_chunk_into: a plane-resident accumulator drains; nothing else does"
                )
            }
        }
    }

    /// [`drained_into`](Tile::drained_into) through the scratch this grid was opened with,
    /// where `dest` is one its fragments must bounce to reach.
    fn drain_as_opened<A: Numeric, Out: Numeric>(&self, dest: &Tile) {
        let in_place = self.casts_in_place_to::<A, Out>();
        let bounces = dest.fragments_bounce(in_place);
        match &self.kind {
            TileKind::PlanePartition(p) => match comptime!(DrainPlan::new(p.held, bounces)) {
                DrainPlan::Straight => drain_below::<A, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Copy),
                ),
                // One slot shared: each spill and add pair off at its leaf.
                DrainPlan::BounceEach => drain_below::<A, Out>(
                    self,
                    dest,
                    comptime!(self.place.below().to_vec()),
                    0usize,
                    comptime!(DrainPass::Bounce),
                ),
                // Own slots: every spill precedes any add, so the descent runs twice.
                DrainPlan::BounceTogether => {
                    self.sync_scratch();
                    drain_below::<A, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Spill),
                    );
                    self.sync_scratch();
                    drain_below::<A, Out>(
                        self,
                        dest,
                        comptime!(self.place.below().to_vec()),
                        0usize,
                        comptime!(DrainPass::Add),
                    );
                    self.sync_scratch();
                }
            },
            // No grid: the accumulator is one tile and the drain is one store.
            TileKind::PlaneTile(_) => drain_leaf::<A, Out>(self, dest, comptime!(DrainPass::Copy)),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::drained_into: a plane-resident accumulator drains; nothing else does")
            }
        }
    }
}

/// Panics unless a warpgroup holds each of the output's tiles: the narrowest scope a level below
/// `place` deals them to, or the one holding `place` where none does. A warpgroup MMA takes all
/// four of its planes, so a plane, a unit or another group size would issue it short.
fn assert_warpgroups_hold(place: &Placement) {
    let holder = place
        .below()
        .iter()
        .filter_map(|level| level.coverage().scope())
        .min()
        .unwrap_or_else(|| place.holder());
    assert!(
        holder == ComputeScope::WARPGROUP,
        "Tile::accumulator: a warpgroup MMA's tile is held by {:?}, its four planes together, \
         and this one's by {holder:?}; deal the tiles to `plane_groups(4, ..)`",
        ComputeScope::WARPGROUP
    );
}

/// Panics where a plane group level splits an axis the output at `place` lacks: each group would
/// hold a partial of the same cells, and nothing merges plane groups' partials yet.
fn assert_groups_own_their_box(place: &Placement) {
    for level in &place.levels {
        let Coverage::Distribute(ComputeScope::PlaneGroup { .. }) = level.coverage() else {
            continue;
        };
        for axis in level.axes() {
            let split = !matches!(level.count(axis), Some(Count::Stated(1)));
            assert!(
                !(level.distributes(axis) && split && !place.space.contains(axis)),
                "Tile::accumulator: plane groups split {axis:?}, which the output lacks, so each \
                 would hold a partial of the same cells, and nothing merges plane groups' \
                 partials yet; deal each group a box of its own"
            );
        }
    }
}

/// A register block `out`'s plane sums `lhs · rhs` into, run under `config`.
#[cube]
fn register_accumulator<EA: Numeric>(
    out: &Tile,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) -> Tile {
    let lw = lhs.vector_size();
    // The block's lines are the rhs's loads, or, where the rhs runs along the contraction, one
    // column's run of a load summed into a cell.
    let rhs_load = rhs.vector_tile();
    let rw = comptime!(rhs_load.run_length());
    let aw = out.vector_size();
    let fold = comptime!(memory::contracted_per_step(
        &lhs.place.space,
        &rhs.place.space,
        &out.place.space,
        lw,
        rw,
        aw
    ));
    // A block outliving the leaf cannot spread lines, so its lines and the sink's agree or fold.
    comptime!(assert!(
        fold > 1 || rw == aw,
        "Tile::accumulator: the block's lines are the rhs's ({rw} wide) and drain into \
         {aw}-wide cells; a stage served wider than its sink is the memory-backed leaf's \
         (Tile::accumulating)"
    ));
    // A block contracting runs along the contraction sums each into its cell, one value wide.
    let width = comptime!(if fold > 1 { 1 } else { rw });
    accumulator_in::<EA>(
        out,
        lhs,
        comptime!(Instruction::Registers { config }),
        width,
        fold,
        comptime!(Accumulation::Contraction(semiring)),
    )
}

/// The plane-resident grid an accumulator gathers in, in `form`, at its identity.
#[cube]
fn accumulator_in<EA: Numeric>(
    out: &Tile,
    lhs: &Tile,
    #[comptime] form: Instruction,
    #[comptime] vector_size: usize,
    #[comptime] fold: usize,
    #[comptime] accumulation: Accumulation,
) -> Tile {
    comptime!(assert!(
        matches!(form, Instruction::Registers { .. })
            || accumulation == Accumulation::Contraction(Semiring::SUM_PROD),
        "Tile::accumulator: a hardware instruction accumulates under the sum of products alone, \
         not {accumulation:?}; contract in a register block to fold under another"
    ));
    let mut acc = PlanePartition::mirror::<EA>(
        comptime!(out.place.space.clone()),
        comptime!(MatrixAxes::accumulator(&out.place.space, &lhs.place.space)),
        comptime!(form),
        comptime!(GridShape::new(&out.place, &lhs.place.space)),
        vector_size,
        fold,
        accumulation,
        comptime!(out.place.depth),
        comptime!(out.place.levels.clone()),
    );
    acc.init_identity::<EA>(comptime!(accumulation.monoid()));
    acc
}

#[cube]
impl Tile {
    /// A plane-resident accumulator over one region of `walk` along `axes`, summing `lhs` times a
    /// factor in the matrix instruction `instruction` names, opened at zero: the accumulator
    /// [`Tile::scratch`] would open, without the memory, for a product contracted again before
    /// anything drains it, as an attention's score.
    pub fn stage_accumulator<EA: Numeric>(
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        lhs: &Tile,
        #[comptime] instruction: Instruction,
    ) -> Tile {
        comptime!(assert!(
            !matches!(instruction, Instruction::Registers { .. }),
            "Tile::stage_accumulator: a register block's accumulator is a unit's, opened on the \
             output it drains into"
        ));
        let place = comptime!(Placement::new(
            walk.level.child(&walk.space).subspace(&axes),
            walk.depth(),
            walk.parent.path.root_levels(),
        ));
        let accumulation = comptime!(Accumulation::Contraction(Semiring::SUM_PROD));
        let mut acc = PlanePartition::mirror::<EA>(
            comptime!(place.space.clone()),
            comptime!(MatrixAxes::accumulator(&place.space, &lhs.place.space)),
            instruction,
            comptime!(GridShape::new(&place, &lhs.place.space)),
            1usize,
            1usize,
            accumulation,
            comptime!(place.depth),
            comptime!(place.levels.clone()),
        );
        acc.init_identity::<EA>(comptime!(accumulation.monoid()));
        acc
    }
}

/// [`Tile::drained_into`]'s descent over `levels[i..]`, applying `pass` at every leaf.
#[cube]
fn drain_below<A: Numeric, O: Numeric>(
    acc: &Tile,
    dest: &Tile,
    #[comptime] levels: Vec<Level>,
    #[comptime] i: usize,
    #[comptime] pass: DrainPass,
) {
    if comptime!(i == levels.len()) {
        drain_leaf::<A, O>(acc, dest, pass);
    } else {
        let level = comptime!(levels[i].clone());
        // A walked level's coordinates must fold to constants: its regions select fragments.
        if comptime!(level.coverage() == Coverage::Walk) {
            for region in dest.over(&level).unrolled() {
                drain_below::<A, O>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                );
            }
        } else {
            for region in dest.over(&level) {
                drain_below::<A, O>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                );
            }
        }
    }
}

/// [`Tile::drained_chunk_into`]'s descent over `levels[i..]`: [`drain_below`]'s, the fragment
/// grid's coordinates summed along the way into `place`, the chunk a fragment falls in being
/// `place` modulo `chunks`.
#[cube]
#[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
fn drain_chunk_below<A: Numeric, O: Numeric>(
    acc: &Tile,
    dest: &Tile,
    #[comptime] levels: Vec<Level>,
    #[comptime] i: usize,
    #[comptime] pass: DrainPass,
    place: usize,
    turn: usize,
    #[comptime] chunks: usize,
) {
    if comptime!(i == levels.len()) {
        drain_chunk_leaf::<A, O>(acc, dest, pass, place % chunks == turn);
    } else {
        let level = comptime!(levels[i].clone());
        let axes = comptime!(dest.place.space.axes().collect::<Vec<_>>());
        // Only the fragment grid, a walked level, places a fragment: every plane holding the same
        // tile reaches its fragments at the same places, the planes' own level adding nothing.
        if comptime!(level.coverage() == Coverage::Walk) {
            for region in dest.over(&level).unrolled() {
                let mut here = place;
                #[unroll]
                for a in 0..comptime!(axes.len()) {
                    here += region.coord(comptime!(axes[a]));
                }
                drain_chunk_below::<A, O>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                    here,
                    turn,
                    chunks,
                );
            }
        } else {
            for region in dest.over(&level) {
                drain_chunk_below::<A, O>(
                    &acc.at(&region),
                    &dest.at(&region),
                    comptime!(levels.clone()),
                    comptime!(i + 1),
                    pass,
                    place,
                    turn,
                    chunks,
                );
            }
        }
    }
}

/// [`drain_leaf`] for a fragment that lands only where `lands`: every barrier is reached
/// whatever it says, so planes landing different fragments still meet at the same ones.
#[cube]
fn drain_chunk_leaf<A: Numeric, O: Numeric>(
    acc: &Tile,
    dest: &Tile,
    #[comptime] pass: DrainPass,
    lands: bool,
) {
    match comptime!(pass) {
        DrainPass::Copy => {
            if lands {
                let mut window = dest.clone();
                window.copy_cast_from(acc);
            }
        }
        DrainPass::Spill => {
            if lands {
                acc.spill_to_scratch::<A>();
            }
        }
        DrainPass::Add => {
            if lands {
                let mut window = dest.clone();
                window.add_from_scratch::<O, A>(acc);
            }
        }
        DrainPass::Bounce => {
            let mut window = dest.clone();
            acc.sync_scratch();
            if lands {
                acc.spill_to_scratch::<A>();
            }
            acc.sync_scratch();
            if lands {
                window.add_from_scratch::<O, A>(acc);
            }
            acc.sync_scratch();
        }
    }
}

/// One tile of a drain against the one window it lands in.
#[cube]
fn drain_leaf<A: Numeric, O: Numeric>(acc: &Tile, dest: &Tile, #[comptime] pass: DrainPass) {
    match comptime!(pass) {
        DrainPass::Copy => {
            let mut window = dest.clone();
            window.copy_cast_from(acc);
        }
        DrainPass::Spill => acc.spill_to_scratch::<A>(),
        DrainPass::Add => {
            let mut window = dest.clone();
            window.add_from_scratch::<O, A>(acc);
        }
        DrainPass::Bounce => {
            let mut window = dest.clone();
            acc.sync_scratch();
            acc.spill_to_scratch::<A>();
            acc.sync_scratch();
            window.add_from_scratch::<O, A>(acc);
            acc.sync_scratch();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{CubeOrder, PlaneOrder};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A plane's `16 × 16` tile, walked through `K`.
    fn plane() -> Levels {
        Levels::leaf(&[(M, 16), (N, 16), (K, 16)]).walk_every(&[K])
    }

    /// A plane's windows sized over its own box: one for every plane of its cube.
    fn windows(levels: Vec<Level>) -> usize {
        let partitioning =
            Partitioning::new(Space::new(&[(M, 64), (N, 64), (K, 64)]), levels.clone());
        plane_windows(&partitioning.box_of(ComputeScope::Plane), &levels)
    }

    /// The planes of the launch, four to a cube, size the four windows a cube level holding four
    /// planes does: a plane's own box holds one tile of the level, not the cube's four.
    #[test]
    fn the_launch_planes_size_a_window_for_every_plane_of_the_cube() {
        let of_each_cube = plane().planes(&[(M, 2), (N, 2)]).cubes(&[M, N]).build();
        let of_the_launch = plane()
            .launch_planes(
                &[M, N],
                PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor),
            )
            .build();
        assert_eq!(windows(of_each_cube), 4);
        assert_eq!(windows(of_the_launch), 4);
    }

    /// Blocks of one plane grouped two to a cube: the cube's two planes, not the block's one.
    #[test]
    fn blocks_grouped_in_a_cube_size_a_window_for_each_of_its_planes() {
        let levels = plane()
            .launch_planes(
                &[M, N],
                PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(2),
            )
            .build();
        assert_eq!(windows(levels), 2);
    }
}

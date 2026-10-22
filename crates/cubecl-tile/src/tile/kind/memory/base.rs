//! What a [`Memory`] is: its [`Backing`], its [`Store`], and how it may be touched ([`Access`]).

use cubecl::prelude::*;

use cubecl_std::tensor::WriteOnly;

use crate::*;

/// A lifetime-erased buffer, its fixed `layout`, and the window this tile looks at. The buffer
/// holds its element as data; what reads or writes it names the element ([`Store::buffer`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Memory {
    /// Which memory the bytes sit in.
    #[cube(comptime)]
    pub(crate) address: AddressSpace,
    /// What the bytes are and mean.
    pub(crate) store: Store,
    /// How a logical coordinate becomes a buffer offset; fixed at construction.
    pub(crate) layout: BufferLayout,
    /// The region of the physical buffer this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// How the tile's logical axes address the buffer's physical ones; fixed at construction.
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// The runtime half of [`projection`](Self::projection): its coefficients and window phase.
    pub(crate) map: RuntimeMap,
    /// The runtime constant terms of the projection, one signed value per dynamic-offset axis.
    pub(crate) offsets: Coords<i32>,
    /// The window origin's line offset, accumulated across [`at`](Tile::at)s.
    pub(crate) window_start: u32,
    /// How this store may be touched; decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What the plane's units are to these cells, stamped across [`at`](Tile::at)s.
    #[cube(comptime)]
    pub(crate) unit_share: UnitShare,
    /// What one instance holds of these cells; read by accumulators only.
    #[cube(comptime)]
    pub(crate) split_share: SplitShare,
    /// What the accumulation being lowered starts from ([`InitFrom`]).
    #[cube(comptime)]
    pub(crate) init_from: InitFrom,
    /// How a contraction into this window runs ([`Tile::accumulating`](crate::Tile::accumulating));
    /// `None` until stated.
    #[cube(comptime)]
    pub(crate) contraction: Option<Contraction>,
    /// Where this tile's cells sit in the buffer they were filled from; `Some` only for a gathered
    /// stage.
    pub(crate) source_window: ComptimeOption<SourceWindow>,
    /// The table these values index ([`Tile::lookup`](crate::Tile::lookup)); empty when none.
    pub(crate) codebook: Codebook,
    /// The table that places this window ([`Indexing`]); `None` for an operand read where its
    /// windows are cut.
    pub(crate) index: ComptimeOption<IndexTable>,
}

/// How a contraction into a memory window runs: the register block a unit's share is tiled into,
/// and the semiring it sums under.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Contraction {
    pub(crate) block: RegisterBlock,
    pub(crate) semiring: Semiring,
}

/// Which memory a [`Memory`] tile's buffer sits in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum AddressSpace {
    Global,
    Shared,
}

/// What backs a [`Memory`]'s values: an addressable buffer or an erased call.
/// Address-shaped operations on a call are a comptime panic.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
#[allow(dead_code)] // Built through the derived `new_*` expand constructors.
pub(crate) enum Backing {
    /// Bytes this kernel addresses directly, grouped into lines at `vector_size`.
    Buffer(Raw<Slice>),
    /// A write-only destination that is not memory (an erased tensor).
    WriteCall(RawCall<WriteOnly>),
    /// A read-only producer that is not memory.
    ReadCall(RawCall<ReadOnly>),
}

/// What a [`Memory`]'s values are: their backing, line width and packing.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Store {
    /// What backs the values.
    pub(crate) backing: Backing,
    /// Physical line size of the destination, `1` when unvectorized.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// How the buffer's values sit in it, from [`TileSpec::packed`].
    #[cube(comptime)]
    pub(crate) packing: Packing,
    /// How the buffer is stored: its stated storage tiles, finest first
    /// ([`TileSpec::stored_tiles`]); empty for a plain buffer and every stage. What one load of it
    /// covers follows from these and the width ([`Tile::vector_tile`]).
    #[cube(comptime)]
    pub(crate) stored_tiles: Vec<(Axis, usize)>,
}

#[cube]
impl Memory {
    /// What one vector load of this memory covers over a window spanning `space`: the stored
    /// tiles that hold [`vector_size`](Store::vector_size) values, a run along the innermost axis
    /// where none are stated ([`VectorTile::new`]).
    pub(crate) fn vector_tile(&self, #[comptime] space: &Space) -> comptime_type!(VectorTile) {
        let width = comptime!(self.store.vector_size);
        comptime!(
            VectorTile::new(
                &self.store.stored_tiles,
                space.axis_at(space.rank() - 1),
                width
            )
            .unwrap_or_else(|why| panic!("Memory: {width} values a load: {why}"))
        )
    }

    /// How a contraction into this window runs, as stated with
    /// [`Tile::accumulating`](crate::Tile::accumulating). Refuses a window never stated.
    pub(crate) fn stated_contraction(&self) -> comptime_type!(Contraction) {
        comptime!(self.contraction.unwrap_or_else(|| panic!(
            "Tile::mma: a memory window contracts under the register block and semiring it is \
             stated with; state them with Tile::accumulating(block, semiring)"
        )))
    }
}

#[cube]
impl Store {
    /// A store whose buffer is laid down in rows, stored in no tiles: every stage.
    pub(crate) fn untiled(
        backing: Backing,
        #[comptime] vector_size: usize,
        #[comptime] packing: Packing,
    ) -> Store {
        Store {
            backing,
            vector_size,
            packing,
            stored_tiles: comptime!(Vec::new()),
        }
    }

    /// Whether these values sit at an address, which a slice-shaped write such as a fragment's
    /// store intrinsic needs; a write call has none.
    pub(crate) fn addressed(&self) -> comptime_type!(bool) {
        match &self.backing {
            Backing::Buffer(_) | Backing::ReadCall(_) => comptime!(true),
            Backing::WriteCall(_) => comptime!(false),
        }
    }

    /// The bytes, for a destination that has an address.
    // `Box<[T]>` is cubecl's owned-slice handle, not a Rust box; `&[T]` is a different kernel type.
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer<T: Numeric>(&self) -> &Box<[T]> {
        match &self.backing {
            Backing::Buffer(buffer) => buffer.at::<T>(),
            Backing::WriteCall(_) => panic!(
                "Store::buffer: this tile's backing is written through a call, which has no \
                 address, it can only be written through its layout"
            ),
            Backing::ReadCall(_) => panic!(
                "Store::buffer: this tile's backing is read through a call, which has no \
                 address, it can only be read through its layout (Memory::read_view), so the \
                 slice-shaped paths (a dense run, a re-typed quant storage, a tensor-map load) \
                 are closed to it"
            ),
        }
    }

    /// Free the shared memory a buffer backing is held in, for what is declared after it to take;
    /// a call holds none.
    pub(crate) fn free(&self) {
        match &self.backing {
            Backing::Buffer(buffer) => buffer.free(),
            Backing::WriteCall(_) | Backing::ReadCall(_) => {}
        }
    }

    /// Whether the values have an address: a buffer, rather than a call.
    pub(crate) fn has_address(&self) -> comptime_type!(bool) {
        match &self.backing {
            Backing::Buffer(_) => comptime!(true),
            Backing::WriteCall(_) | Backing::ReadCall(_) => comptime!(false),
        }
    }

    /// The mutable twin of [`buffer`](Self::buffer).
    #[allow(clippy::borrowed_box)]
    pub(crate) fn buffer_mut<T: Numeric>(&mut self) -> &mut Box<[T]> {
        match &mut self.backing {
            Backing::Buffer(buffer) => buffer.at_mut::<T>(),
            Backing::WriteCall(_) => panic!(
                "Store::buffer_mut: this tile's backing is written through a call, which has no \
                 address, it can only be written through its layout"
            ),
            Backing::ReadCall(_) => panic!(
                "Store::buffer_mut: this tile's backing is read through a call, which is \
                 read-only"
            ),
        }
    }
}

/// How a [`Memory`] may be touched; plain comptime data.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Access {
    /// Whether the window still covers the whole buffer.
    pub whole: bool,
    pub overhang: Overhang,
    /// What a write here does to the cell it lands on.
    pub write: Write,
    /// The units that share a cooperative fill of this window.
    pub fill: FillUnits,
    /// What the storage tiles are to this window.
    pub storage: WindowStorage,
    /// Who moves this tile's lines into a stage filled from it. Stated by the operand's spec and
    /// carried down its windows; a stage copied onward is copied by its units.
    pub delivery: Delivery,
}

/// The units that share a cooperative fill of a window: every unit of the cube, the units of the
/// planes the cube sets aside to fill its stages, or the units of one plane, for a stage that
/// plane owns and fills alone ([`Stages::smem`](crate::Stages::smem)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FillUnits {
    /// Whose units they are: the cube's, or one plane's.
    pub scope: ComputeScope,
    /// How many there are, `0` when unknown: a stage filled from this tile emits its fill
    /// straight-line when it knows. The launch's cube size, or one plane's width.
    pub count: usize,
    /// The planes at the end of the cube set aside to fill its stages, whose units alone share
    /// the fill of a cube's stage; `0` where every unit of the cube does.
    pub fillers: usize,
}

impl FillUnits {
    /// Every unit of the cube, `count` of them (`0` when unknown).
    pub(crate) fn cube(count: usize) -> Self {
        FillUnits {
            scope: ComputeScope::Cube,
            count,
            fillers: 0,
        }
    }

    /// The units of the `fillers` planes at the end of the cube set aside to fill its stages
    /// ([`Level::fillers`](crate::Level::fillers)), how many unknown: the launch states the
    /// cube's units, not the plane's width.
    pub(crate) fn filling_planes(fillers: usize) -> Self {
        FillUnits {
            scope: ComputeScope::Cube,
            count: 0,
            fillers,
        }
    }

    /// These units narrowed to `holder`'s where it has fewer: a plane's window of a buffer the
    /// cube fills is the plane's to fill.
    pub(crate) fn held_by(self, holder: ComputeScope) -> Self {
        match holder < self.scope {
            true => FillUnits {
                scope: holder,
                count: 0,
                fillers: 0,
            },
            false => self,
        }
    }
}

#[cube]
impl FillUnits {
    /// This unit's position among the units `fill` names, which a cooperative fill takes its
    /// lines at: its position in the cube, among the filling planes' units, or in its plane.
    ///
    /// A plane is the launch's `x` ([`Partitioning::cube_dim`]), so a unit's position in its plane
    /// is `UNIT_POS_X` and the plane's width `CUBE_DIM_X`; the filling planes are the cube's last.
    pub(crate) fn worker(#[comptime] fill: FillUnits) -> usize {
        comptime!(assert!(
            fill.scope != ComputeScope::Unit,
            "FillUnits::worker: a cooperative fill is shared by a cube or a plane, never one unit"
        ));
        if comptime!(fill.fillers > 0) {
            (UNIT_POS - (CUBE_DIM - comptime!(fill.fillers as u32) * CUBE_DIM_X)) as usize
        } else {
            ComputeScope::unit(comptime!(fill.scope))
        }
    }

    /// How many units `fill` names at runtime: the cube's, the filling planes', or one plane's
    /// ([`worker`](FillUnits::worker)).
    pub(crate) fn workers(#[comptime] fill: FillUnits) -> usize {
        if comptime!(fill.fillers > 0) {
            return comptime!(fill.fillers) * CUBE_DIM_X as usize;
        }
        match comptime!(fill.scope) {
            ComputeScope::Cube => CUBE_DIM as usize,
            ComputeScope::Plane => CUBE_DIM_X as usize,
            ComputeScope::PlaneGroup { planes } => planes * CUBE_DIM_X as usize,
            ComputeScope::Unit => comptime!(panic!(
                "FillUnits::workers: a cooperative fill is shared by a cube or a plane, never \
                 one unit"
            )),
        }
    }
}

/// What a write to a store does to the cell it lands on; stated by the binding operand. `Replace`
/// is the writer's own cell, `Accumulate` adds atomically so writers need not know of each other,
/// `Exclusive` adds with a plain read and write, one writer at a cell at a time in the order its
/// [`Schedule`] sets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Write {
    /// Replaces the cell.
    Replace,
    /// Adds into the cell, atomically: the writers' partials land in any order.
    Accumulate,
    /// Replaces the cell on its first writer's turn and adds into it, reading it and writing the
    /// sum back a line at a time, on every later one. One writer reaches a cell at a time, in the
    /// order the schedule sets, so the sum is the same bits from run to run.
    Exclusive(Schedule),
}

/// The order a destination's [`Exclusive`](Write::Exclusive) writers take their turns at a cell in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Schedule {
    /// One writer at a time over the whole box, in the order of their runs, handed on through a
    /// counter: the cubes of a split taking turns at a [`Relay`](crate::launch::Relay)'s carry.
    Sequential,
    /// Every writer every round, each on its own chunk of the box: in round `r`, writer `w` takes
    /// chunk `(w + r) mod writers`, a cube barrier between rounds. The planes of one cube meeting
    /// in shared memory ([`SmemCyclicAccumulation`](crate::kind::SmemCyclicAccumulation)), where a
    /// barrier orders them.
    Cyclic,
}

impl Write {
    /// Refuse an accumulation `split` leaves in pieces unless this write adds them, a relay whose
    /// partials the planes of one cube write at once, which no turn across cubes serializes, and
    /// a `monoid` other than a sum into a destination that adds.
    pub(crate) fn admits(self, split: SplitShare, monoid: Monoid, site: &str) {
        match (self, monoid) {
            (Write::Accumulate | Write::Exclusive(_), Monoid::Prod | Monoid::Max | Monoid::Min) => {
                panic!(
                    "{site}: this destination adds ({self:?}), atomically or relayed, so a \
                 {monoid:?} accumulation into it would come out summed. Drain it into a \
                 replacing destination, combining the partials first where they are split."
                )
            }
            (Write::Accumulate | Write::Exclusive(_), Monoid::Sum) | (Write::Replace, _) => {}
        }
        match (split, self) {
            (SplitShare::Whole, _)
            | (
                SplitShare::PartialAcrossCubes | SplitShare::PartialAcrossPlanes,
                Write::Accumulate,
            )
            | (SplitShare::PartialAcrossCubes, Write::Exclusive(Schedule::Sequential))
            | (SplitShare::PartialAcrossPlanes, Write::Exclusive(Schedule::Cyclic)) => {}
            (SplitShare::PartialAcrossCubes, Write::Exclusive(Schedule::Cyclic)) => panic!(
                "{site}: this accumulator's cells are split across cubes and its destination \
                 takes its writers in cyclic rounds, which a cube barrier orders: it orders the \
                 planes of one cube, never cubes. Take the cubes' turns through a `Relay`, or \
                 combine them atomically."
            ),
            (SplitShare::PartialAcrossPlanes, Write::Exclusive(Schedule::Sequential)) => panic!(
                "{site}: this accumulator's cells are split across the planes of one cube and its \
                 destination is relayed, a plain read and write taken one cube at a time; the planes \
                 would write the same line at once and lose each other's partial. \
                 Drain into an atomic destination (bind it as an `AccumulateArg`), or combine \
                 the planes' partials in the cube before its turn at the relay, atomically or in \
                 cyclic rounds (`smem_cyclic_accumulation`)."
            ),
            (SplitShare::PartialAcrossCubes | SplitShare::PartialAcrossPlanes, Write::Replace) => {
                panic!(
                    "{site}: this accumulator's cells are split across planes or cubes and its \
                 destination replaces rather than accumulates, so every partial but one would be \
                 lost. \
                 A contracted axis distributed across planes or cubes gives each instance a \
                 slice of the contraction, and none of them holds a whole cell. \
                 Drain into an accumulating destination (bind it as an `AccumulateArg`), \
                 distribute the contraction across the plane's units instead \
                 (`distribute(units(n), ..)`, combined in the plane's registers), or give the \
                 output an axis of its own for the split."
                )
            }
        }
    }
}

/// How a store relates to the window overhanging its valid data (past [`Window`]'s `bound`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Overhang {
    /// Impossible: the buffer is allocated to exactly the tile (smem).
    Never,
    /// Excluded at launch: every shape divides its tiling (unchecked gmem).
    Fits,
    /// Reads/writes past `bound` are masked per the window's [`Boundary`].
    Masked,
    /// Masked past the end of a run ([`Memory::within`]) and nowhere else: a stage the walk proves
    /// inside the run reads it as [`Fits`](Overhang::Fits).
    Run,
}

/// Boundary handling mode for out-of-bounds reads/writes, carried by `Window`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Boundary {
    /// Out-of-bounds reads return zero; writes are skipped.
    Zero,
    /// Out-of-bounds reads/writes clamp to the edge cell.
    Clamp,
}

impl Overhang {
    /// The flag a [`Masked`] is built with.
    pub(crate) fn masks(&self) -> bool {
        matches!(self, Overhang::Masked | Overhang::Run)
    }
}

/// Whether a read still proves its own bounds, stated by the reader.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Guard {
    /// Mask the overhang and apply the window's [`Boundary`] on every access.
    Checked,
    /// The reader proved its whole box is in bounds; reading past it is out of bounds, not masked.
    Proved,
}

impl Guard {
    /// Whether this guard still costs a test per access.
    pub(crate) fn checks(self) -> bool {
        matches!(self, Guard::Checked)
    }
}

/// How an operand's buffer is stored: what a launched spec states.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Storage {
    /// Untiled storage: every window lies inside the one storage tile.
    Strided,
    /// Storage-tiled, and the window may span several tiles.
    /// The level is the one whose tile the storage tile is; `None` if no level's.
    Tiled(Option<usize>),
}

/// What a storage tile is to the window an operand is read through.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum WindowStorage {
    /// As the buffer is stored.
    Stored(Storage),
    /// Storage-tiled and inside one storage tile: one contiguous run from its origin.
    Contiguous,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A relay takes one cube at a time: the cubes of a split meet it in turns, and an atomic
    /// destination takes a split at either scope.
    #[test]
    fn a_relay_takes_a_split_across_cubes() {
        Write::Exclusive(Schedule::Sequential).admits(
            SplitShare::PartialAcrossCubes,
            Monoid::Sum,
            "test",
        );
        Write::Accumulate.admits(SplitShare::PartialAcrossCubes, Monoid::Sum, "test");
        Write::Accumulate.admits(SplitShare::PartialAcrossPlanes, Monoid::Sum, "test");
    }

    /// The planes of one cube write a cell at once, which no turn across cubes serializes.
    #[test]
    #[should_panic(expected = "split across the planes of one cube")]
    fn a_relay_refuses_a_split_across_planes() {
        Write::Exclusive(Schedule::Sequential).admits(
            SplitShare::PartialAcrossPlanes,
            Monoid::Sum,
            "test",
        );
    }

    /// A destination that adds sums whatever it is handed, so it takes only a sum.
    #[test]
    #[should_panic(expected = "would come out summed")]
    fn an_atomic_destination_refuses_a_max() {
        Write::Accumulate.admits(SplitShare::Whole, Monoid::Max, "test");
    }

    #[test]
    #[should_panic(expected = "would come out summed")]
    fn a_relay_refuses_a_max() {
        Write::Exclusive(Schedule::Sequential).admits(
            SplitShare::PartialAcrossCubes,
            Monoid::Max,
            "test",
        );
    }

    /// A replacing destination takes any monoid: each cell is its writer's own.
    #[test]
    fn a_replacing_destination_takes_any_monoid() {
        for monoid in [Monoid::Sum, Monoid::Prod, Monoid::Max, Monoid::Min] {
            Write::Replace.admits(SplitShare::Whole, monoid, "test");
        }
    }
}

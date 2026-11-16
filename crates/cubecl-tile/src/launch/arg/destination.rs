//! The [`Destination`] seam, through which a kernel stores into [`Output`] or a caller's own
//! argument, or hands each output tile it computes to the op that reads it.

use cubecl::prelude::*;

use crate::{
    Bound, Output, OutputArgs, Partitioning, Refusal, Region, Tile, TileExpand, TileTensor,
    TileTensorLaunch, kind::Write, launch::Unbound,
};

/// Which launch argument carries a kernel's output and how the kernel serves it: a storing
/// destination serves the output as a [`Tile`], one that carries a state across the output tiles
/// is the next op fused into the routine.
#[cube]
pub trait Destination: Send + Sync + 'static {
    /// The launch argument carrying the output and its spec, at element `E`.
    type Arg<E: Numeric>: LaunchArg + CubeType;
    /// The destination as served to the kernel, built once by [`serve`](Self::serve).
    type Served<E: Numeric>: ServedDestination;

    /// Build the destination for the kernel, once, under its one `partitioning`, storing in lines
    /// of `V` where the argument does not state its own width.
    fn serve<E: Numeric, V: Size>(
        arg: &Self::Arg<E>,
        partitioning: &Partitioning,
    ) -> Self::Served<E>;
}

/// A [`Destination`] as served to a kernel, from the start of the kernel to its end.
#[cube]
pub trait ServedDestination: CubeType {
    /// What an opened destination carries across the output tiles an owner region computes.
    type Opened: OpenedDestination;

    /// A tile over the routine's output space, holding no data: where the routine opens its
    /// accumulators. A storing destination's is where the output lies.
    fn tile(&self) -> Tile;

    /// Open the destination once per owner region the routine chooses (a unit, a plane or a
    /// cube), before its first output tile. `sum` is laid out as every sum
    /// [`consume`](OpenedDestination::consume) will receive, which a state kept per slice follows.
    fn open(&self, owner: &Region, sum: &Tile) -> Self::Opened;

    /// Called once by every unit of every launched cube, in uniform control flow, after every
    /// [`close`](OpenedDestination::close) and its last use of the tile, with no unit exited, so
    /// a destination may synchronize the cube and use plane operations here. Stores made through
    /// the tile are issued but not known to be visible to any other unit; making them visible is
    /// the destination's job where it needs it.
    /// A TMA store has only been read out of shared memory: its global writes are not known to be
    /// complete here.
    fn finish(&self);
}

/// A [`ServedDestination`] opened for one owner region, from its first output tile to its last.
#[cube]
pub trait OpenedDestination: CubeType {
    /// Consume the output tile at `at`, once per finished tile, in the order the owner computes
    /// them, possibly in divergent control flow, so it may not use plane operations or
    /// synchronize unless the routine says otherwise. `sum` may carry scales
    /// ([`Tile::scaled`]) that a draining reader applies and any other reader must.
    fn consume(&mut self, at: &Region, sum: &Tile);

    /// Close once per [`open`](ServedDestination::open), after the last
    /// [`consume`](Self::consume) and before [`finish`](ServedDestination::finish), in the
    /// control flow of the level it was opened at: a cube-level close may synchronize the cube.
    fn close(&mut self);
}

/// A tile is the destination of a store: it opens to itself, and each sum drains where it lies.
#[cube]
impl ServedDestination for Tile {
    type Opened = Tile;

    fn tile(&self) -> Tile {
        self.clone()
    }

    fn open(&self, _owner: &Region, _sum: &Tile) -> Tile {
        self.clone()
    }

    fn finish(&self) {}
}

#[cube]
impl OpenedDestination for Tile {
    fn consume(&mut self, at: &Region, sum: &Tile) {
        sum.drained_into(&self.at(at));
    }

    fn close(&mut self) {}
}

/// [`Destination`]'s host half: how what the launch built becomes the argument.
pub trait DestinationLaunch: Destination {
    /// What the argument is made from, before the kernel's element types it.
    type Operand;

    fn arg<E: Numeric>(operand: Self::Operand) -> <Self::Arg<E> as LaunchArg>::RuntimeArg;
}

/// A kernel prepared to store through a [`Destination`] the caller builds afterwards, at
/// [`width`](Self::width): the caller binds the output only once it knows the kernel's spec.
pub trait SinkLaunch {
    /// The width the kernel hands the destination its lines in.
    fn width(&self) -> usize;

    /// The dense output's element count, which a destination standing for it must cover exactly.
    fn elements(&self) -> usize;

    /// Launch through the destination `O`, `sink` building the output's argument from its derived
    /// spec and geometry.
    fn launch<O: DestinationLaunch>(self, sink: impl FnOnce(Unbound) -> O::Operand);
}

/// What a kernel named by a value rather than a type is handed to, once prepared for a sink: each
/// value prepares a launch of its own type, so what receives it takes any [`SinkLaunch`].
pub trait SinkLaunchVisitor {
    type Output;

    /// Receive the launch the value named, or why it could not be prepared.
    fn visit<L: SinkLaunch>(self, prepared: Result<L, Refusal>) -> Self::Output;
}

/// The default [`Destination`]: an [`Output`] buffer.
pub struct Buffered;

#[cube]
impl Destination for Buffered {
    type Arg<E: Numeric> = Output<'static, E>;
    type Served<E: Numeric> = Tile;

    fn serve<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        arg.tile(partitioning)
    }
}

/// [`Write::Replace`] binds an [`Output::Tensor`], [`Write::Accumulate`] an [`Output::Atomic`].
impl DestinationLaunch for Buffered {
    type Operand = (Bound, Write);

    fn arg<E: Numeric>((bound, write): (Bound, Write)) -> OutputArgs<'static, E> {
        match write {
            Write::Replace => bound.output(),
            Write::Accumulate => bound.atomic(),
            Write::Exclusive(_) => panic!(
                "Buffered: a relayed output is opened in the kernel with `TileTensor::relay`, \
                 which takes its turn counters beside it; bind the carry as a plain `TileTensor`"
            ),
        }
    }
}

/// A [`Destination`] the TMA engine stores: the output is a tensor map, served as a tile a
/// shared-memory box is bulk-stored into ([`Tile::smem_box`], [`Tile::copy_from`]), a box at a
/// time, which the descriptor clips at the tensor's edge. Its width is the descriptor's cell, so
/// a launch binds it one element wide.
pub struct TmaStored;

#[cube]
impl Destination for TmaStored {
    type Arg<E: Numeric> = TileTensor<'static>;
    type Served<E: Numeric> = Tile;

    fn serve<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        arg.tile(partitioning)
    }
}

impl DestinationLaunch for TmaStored {
    type Operand = TileTensorLaunch;

    fn arg<E: Numeric>(operand: TileTensorLaunch) -> TileTensorLaunch {
        operand
    }
}

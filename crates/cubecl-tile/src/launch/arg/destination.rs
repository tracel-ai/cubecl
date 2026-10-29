//! The [`Destination`] seam, through which a kernel stores into [`Output`] or a caller's own
//! argument.

use cubecl::prelude::*;

use crate::{
    Bound, Output, OutputArgs, Partitioning, Refusal, Tile, TileArg, TileArgLaunch, TmaOperand,
    kind::Write, launch::Unbound,
};

/// Which launch argument carries a kernel's output and how the kernel serves it as a [`Tile`].
#[cube]
pub trait Destination: Send + Sync + 'static {
    /// The launch argument carrying the output and its spec, at element `E`.
    type Arg<E: Numeric>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`] under the kernel's one `partitioning`, stored in lines of
    /// `V` where the argument does not state its own width.
    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile;
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

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
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
                "Buffered: a relayed output is opened in the kernel with `TileArg::relay`, which \
                 takes its turn counters beside it; bind the carry as a plain `TileArg`"
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
    type Arg<E: Numeric> = TileArg<'static>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        arg.tile(partitioning)
    }
}

impl DestinationLaunch for TmaStored {
    type Operand = TmaOperand;

    fn arg<E: Numeric>(operand: TmaOperand) -> TileArgLaunch {
        operand.arg()
    }
}

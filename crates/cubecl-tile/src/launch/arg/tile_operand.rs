//! The [`TileOperand`] seam, through which a kernel serves an operand as a tile from a buffer or
//! from a caller's own argument.

use cubecl::prelude::*;

use crate::{Partitioning, Tile};

/// Which launch argument carries an operand and how a kernel serves it as a [`Tile`]: a buffer
/// read and written where it lies ([`StoredTileOperand`](crate::launch::StoredTileOperand)), or a
/// caller's argument whose reads and writes run the caller's code, built on
/// [`GlobalOperand::source`](crate::kind::GlobalOperand::source) or
/// [`GlobalOperand::sink`](crate::kind::GlobalOperand::sink).
///
/// The launch hands the kernel the argument's runtime half, `<Arg as LaunchArg>::RuntimeArg`.
#[cube]
pub trait TileOperand: Send + Sync + 'static {
    /// The launch argument carrying the operand.
    type Arg: LaunchArg + CubeType;

    /// The operand as a tile of `E` under the kernel's one `partitioning`, at the width the
    /// argument states.
    fn tile<E: Numeric>(arg: &Self::Arg, partitioning: &Partitioning) -> Tile;
}

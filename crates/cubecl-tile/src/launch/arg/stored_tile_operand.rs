//! The default [`TileOperand`]: a [`TileTensor`].

use cubecl::prelude::*;

use crate::{Partitioning, Tile, TileTensor, launch::TileOperand};

/// A [`TileOperand`] stored in a buffer, read and written where it lies: a [`TileTensor`], whose
/// runtime half is a [`TileTensorLaunch`](crate::TileTensorLaunch).
pub struct StoredTileOperand;

#[cube]
impl TileOperand for StoredTileOperand {
    type Arg = TileTensor<'static>;

    fn tile<E: Numeric>(arg: &TileTensor<'static>, partitioning: &Partitioning) -> Tile {
        arg.tile(partitioning)
    }
}

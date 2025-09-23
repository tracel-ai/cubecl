use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::CubeOption;

use crate::components::tile::StridedTile;

/// Kind (family) of the tiles returned by a tile reader and ingested by a tile matmul reader
pub trait TileKind: CubeType + Send + Sync + 'static {
    /// Concrete tile instantiated with the element type
    type Tile<E: Numeric>: CubeType;
}

/// Tile is a slice of memory with a stride
#[derive(CubeType)]
pub struct Strided {}

/// Tile is a single value that gets filled in everywhere
#[derive(CubeType)]
pub struct Filled {}

impl TileKind for Strided {
    type Tile<E: Numeric> = StridedTile<E>;
}

impl TileKind for Filled {
    type Tile<E: Numeric> = E;
}

impl<Inner: TileKind> TileKind for CubeOption<Inner> {
    type Tile<E: Numeric> = CubeOption<Inner::Tile<E>>;
}

/// A tile matmul reader, with a specific tile kind
pub trait StageReader {
    /// The kind of the tile used as an input for the tile reader
    type TileKind: TileKind;
}

/// The concrete tile type for a given reader and element type
pub type StageTile<L, E> = <<L as StageReader>::TileKind as TileKind>::Tile<E>;
/// The tile kind of a given reader
pub type StageKind<L> = <L as StageReader>::TileKind;

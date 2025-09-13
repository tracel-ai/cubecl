use cubecl::prelude::*;
use cubecl_core as cubecl;

use cubecl_std::CubeOption;

use crate::components::tile::Tile;

pub trait TileKind: CubeType + Send + Sync + 'static {
    type Tile<E: Numeric>: CubeType;
}

/// Tile is a slice of memory with a stride
#[derive(CubeType)]
pub struct Strided {}

/// Tile is a single value that gets filled in everywhere
#[derive(CubeType)]
pub struct Filled {}

impl TileKind for Strided {
    type Tile<E: Numeric> = Tile<E>;
}

impl TileKind for Filled {
    type Tile<E: Numeric> = E;
}

impl<Inner: TileKind> TileKind for CubeOption<Inner> {
    type Tile<E: Numeric> = CubeOption<Inner::Tile<E>>;
}

pub trait Loader {
    type TileKind: TileKind;
}

pub type LoaderTile<L, E> = <<L as Loader>::TileKind as TileKind>::Tile<E>;
pub type LoaderKind<L> = <L as Loader>::TileKind;

use crate::components::tile::TileConfig;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyTileConfig {}

impl TileConfig for DummyTileConfig {}

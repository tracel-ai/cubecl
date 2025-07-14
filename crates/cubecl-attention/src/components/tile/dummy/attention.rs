use crate::components::{
    AttentionPrecision,
    tile::{TileAttention, dummy::config::DummyTileConfig},
};

pub struct DummyTileAttention {}

impl<AP: AttentionPrecision> TileAttention<AP> for DummyTileAttention {
    type Config = DummyTileConfig;
}

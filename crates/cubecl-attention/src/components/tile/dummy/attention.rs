use std::marker::PhantomData;

use crate::components::{
    AttentionPrecision,
    tile::{TileAttention, dummy::config::DummyTileConfig},
};

pub struct DummyTileAttention<AP: AttentionPrecision> {
    _phantom: PhantomData<AP>,
}

impl<AP: AttentionPrecision> TileAttention<AP> for DummyTileAttention<AP> {
    type Config = DummyTileConfig;
}

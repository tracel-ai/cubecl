use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::tile::TileAttention;

#[derive(CubeType)]
/// Key and Value inputs to the Tile Attention
///
/// Key and Value share the same trait because they may
/// be the same reused underlying fragment
pub enum KeyValueTile<AP: AttentionPrecision, FA: TileAttention<AP>> {
    Reuse(ReuseKV<AP, FA>),
    Key(Key<AP, FA>),
    Value(Value<AP, FA>),
}

#[cube]
impl<AP: AttentionPrecision, FA: TileAttention<AP>> KeyValueTile<AP, FA> {
    pub fn new_key_value(#[comptime] config: FA::Config) -> Self {
        Self::new_Reuse(ReuseKV::new(config))
    }

    pub fn new_key(#[comptime] config: FA::Config) -> Self {
        Self::new_Key(Key::new(config))
    }

    pub fn new_value(#[comptime] config: FA::Config) -> Self {
        Self::new_Value(Value::new(config))
    }

    /// Get the underlying key as readable
    pub fn key(&self) -> &FA::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueTile::Key(key) => &key.fragment,
            KeyValueTile::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    /// Get the underlying key as writable
    pub fn key_mut(&mut self) -> &mut FA::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueTile::Key(key) => &mut key.fragment,
            KeyValueTile::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    /// Get the underlying value as readable
    pub fn value(&self) -> &FA::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueTile::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueTile::Value(value) => &value.fragment,
        }
    }

    /// Get the underlying value as writable
    pub fn value_mut(&mut self) -> &mut FA::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueTile::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueTile::Value(value) => &mut value.fragment,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseKV<AP: AttentionPrecision, FA: TileAttention<AP>> {
    pub fragment: FA::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, FA: TileAttention<AP>> ReuseKV<AP, FA> {
    pub fn new(#[comptime] config: FA::Config) -> Self {
        let fragment = FA::allocate_key_value(config);
        ReuseKV::<AP, FA> { fragment }
    }
}

#[derive(CubeType)]
pub struct Key<AP: AttentionPrecision, FA: TileAttention<AP>> {
    pub fragment: FA::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, FA: TileAttention<AP>> Key<AP, FA> {
    pub fn new(#[comptime] config: FA::Config) -> Self {
        Key::<AP, FA> {
            fragment: FA::allocate_key(config),
        }
    }
}

#[derive(CubeType)]
pub struct Value<AP: AttentionPrecision, FA: TileAttention<AP>> {
    pub fragment: FA::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, FA: TileAttention<AP>> Value<AP, FA> {
    pub fn new(#[comptime] config: FA::Config) -> Self {
        Value::<AP, FA> {
            fragment: FA::allocate_value(config),
        }
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::fragment::AttentionMatmul;

#[derive(CubeType)]
/// Key and Value inputs to the Tile Attention
///
/// Key and Value share the same trait because they may
/// be the same reused underlying fragment
pub enum KeyValueTile<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    Reuse(ReuseKV<AP, AM>),
    Key(Key<AP, AM>),
    Value(Value<AP, AM>),
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> KeyValueTile<AP, AM> {
    pub fn new_key_value(#[comptime] config: AM::Config) -> Self {
        Self::new_Reuse(ReuseKV::new(config))
    }

    pub fn new_key(#[comptime] config: AM::Config) -> Self {
        Self::new_Key(Key::new(config))
    }

    pub fn new_value(#[comptime] config: AM::Config) -> Self {
        Self::new_Value(Value::new(config))
    }

    /// Get the underlying key as readable
    pub fn key(&self) -> &AM::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueTile::Key(key) => &key.fragment,
            KeyValueTile::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    /// Get the underlying key as writable
    pub fn key_mut(&mut self) -> &mut AM::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueTile::Key(key) => &mut key.fragment,
            KeyValueTile::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    /// Get the underlying value as readable
    pub fn value(&self) -> &AM::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueTile::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueTile::Value(value) => &value.fragment,
        }
    }

    /// Get the underlying value as writable
    pub fn value_mut(&mut self) -> &mut AM::KeyValue {
        match self {
            KeyValueTile::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueTile::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueTile::Value(value) => &mut value.fragment,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseKV<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> ReuseKV<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        let fragment = AM::allocate_key_value(config);
        ReuseKV::<AP, AM> { fragment }
    }
}

#[derive(CubeType)]
pub struct Key<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> Key<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        Key::<AP, AM> {
            fragment: AM::allocate_key(config),
        }
    }
}

#[derive(CubeType)]
pub struct Value<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    pub fragment: AM::KeyValue,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> Value<AP, AM> {
    pub fn new(#[comptime] config: AM::Config) -> Self {
        Value::<AP, AM> {
            fragment: AM::allocate_value(config),
        }
    }
}

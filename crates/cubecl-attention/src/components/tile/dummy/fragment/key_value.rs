use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::attention_types::KVT;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::{KeyValueTile, KeyValueTileExpand};

#[derive(CubeType)]
pub enum KeyValueFragment<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    Reuse(ReuseKV<AP, AM>),
    Key(Key<AP, AM>),
    Value(Value<AP, AM>),
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> KeyValueFragment<AP, AM> {
    pub fn new_key_value(#[comptime] config: AM::Config) -> Self {
        Self::new_Reuse(ReuseKV::new(config))
    }

    pub fn new_key(#[comptime] config: AM::Config) -> Self {
        Self::new_Key(Key::new(config))
    }

    pub fn new_value(#[comptime] config: AM::Config) -> Self {
        Self::new_Value(Value::new(config))
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> KeyValueTile<KVT<AP>>
    for KeyValueFragment<AP, AM>
{
    type Key = AM::KeyValue;
    type Value = AM::KeyValue;

    fn key(&self) -> &AM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Key(key) => &key.fragment,
            KeyValueFragment::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    fn key_mut(&mut self) -> &mut AM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Key(key) => &mut key.fragment,
            KeyValueFragment::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    fn value(&self) -> &AM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueFragment::Value(value) => &value.fragment,
        }
    }

    fn value_mut(&mut self) -> &mut AM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueFragment::Value(value) => &mut value.fragment,
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

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::dummy::{FlashMatmul, FlashPrecision};

#[derive(CubeType)]
pub enum KeyValueFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    Reuse(ReuseKV<FP, FM>),
    Key(Key<FP, FM>),
    Value(Value<FP, FM>),
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> KeyValueFragment<FP, FM> {
    pub fn new_key_value(#[comptime] config: FM::Config) -> Self {
        Self::new_Reuse(ReuseKV::new(config))
    }

    pub fn new_key(#[comptime] config: FM::Config) -> Self {
        Self::new_Key(Key::new(config))
    }

    pub fn new_value(#[comptime] config: FM::Config) -> Self {
        Self::new_Value(Value::new(config))
    }

    pub fn key(&self) -> &FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Key(key) => &key.fragment,
            KeyValueFragment::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    pub fn key_mut(&mut self) -> &mut FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Key(key) => &mut key.fragment,
            KeyValueFragment::Value(_) => panic!("Tried to access key on value-only fragment"),
        }
    }

    pub fn value(&self) -> &FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueFragment::Value(value) => &value.fragment,
        }
    }

    pub fn value_mut(&mut self) -> &mut FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Key(_) => panic!("Tried to access value on key-only fragment"),
            KeyValueFragment::Value(value) => &mut value.fragment,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseKV<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub fragment: FM::KeyValue,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> ReuseKV<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        let fragment = FM::allocate_key_value(config);
        ReuseKV::<FP, FM> { fragment }
    }
}

#[derive(CubeType)]
pub struct Key<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub fragment: FM::KeyValue,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> Key<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        Key::<FP, FM> {
            fragment: FM::allocate_key(config),
        }
    }
}

#[derive(CubeType)]
pub struct Value<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub fragment: FM::KeyValue,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> Value<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        Value::<FP, FM> {
            fragment: FM::allocate_value(config),
        }
    }
}

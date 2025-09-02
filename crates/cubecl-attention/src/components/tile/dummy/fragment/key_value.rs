use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::dummy::{FlashMatmul, FlashMatmulConfig as _, FlashPrecision};

#[derive(CubeType)]
pub enum KeyValueFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    Reuse(ReuseKV<FP, FM>),
    Separate(SeparateKV<FP, FM>),
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> KeyValueFragment<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        match config.reuse_key_value() {
            true => Self::new_Reuse(ReuseKV::new(config)),
            false => Self::new_Separate(SeparateKV::new(config)),
        }
    }

    pub fn key(&self) -> &FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &separate_kv.key,
        }
    }

    pub fn key_mut(&mut self) -> &mut FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.key,
        }
    }

    pub fn value(&self) -> &FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &separate_kv.value,
        }
    }

    pub fn value_mut(&mut self) -> &mut FM::KeyValue {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.value,
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
        comment!("Allocating key-value (reuse)");
        let fragment = FM::allocate_key_value(config);
        ReuseKV::<FP, FM> { fragment }
    }
}

#[derive(CubeType)]
pub struct SeparateKV<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    pub key: FM::KeyValue,
    pub value: FM::KeyValue,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> SeparateKV<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        comment!("Allocating key-value (separate)");
        let key = FM::allocate_key(config);
        let value = FM::allocate_value(config);
        SeparateKV::<FP, FM> { key, value }
    }
}

use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::tile::{ScoreMatmul, TileAttentionConfig, ValueMatmul};

#[derive(CubeType)]
pub enum KeyValueFragment<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    Reuse(ReuseKV<AP, SM, VM>),
    Separate(SeparateKV<AP, SM, VM>),
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>>
    KeyValueFragment<AP, SM, VM>
{
    pub fn new<T: TileAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>>(
        #[comptime] config: T,
    ) -> Self {
        match config.reuse_key_value() {
            true => Self::new_Reuse(ReuseKV::new(config.score_config(), config.value_config())),
            false => Self::new_Separate(SeparateKV::new(
                config.score_config(),
                config.value_config(),
            )),
        }
    }

    pub fn key(&self) -> &SM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &separate_kv.key,
        }
    }

    pub fn key_mut(&mut self) -> &mut SM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.key,
        }
    }

    pub fn value(&self) -> &VM::Rhs {
        match self {
            KeyValueFragment::Reuse(_reuse_kv) => comptime!(todo!()),
            KeyValueFragment::Separate(separate_kv) => &separate_kv.value,
        }
    }

    pub fn value_mut(&mut self) -> &mut VM::Rhs {
        match self {
            KeyValueFragment::Reuse(_reuse_kv) => comptime!(todo!()),
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.value,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseKV<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    pub fragment: SM::Rhs,
    #[cube(comptime)]
    _phantom: PhantomData<VM>,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> ReuseKV<AP, SM, VM> {
    pub fn new(
        #[comptime] score_config: SM::Config,
        #[comptime] _value_config: VM::Config,
    ) -> Self {
        let fragment = SM::allocate_rhs(score_config);
        ReuseKV::<AP, SM, VM> {
            fragment,
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType)]
pub struct SeparateKV<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    pub key: SM::Rhs,
    pub value: VM::Rhs,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> SeparateKV<AP, SM, VM> {
    pub fn new(#[comptime] score_config: SM::Config, #[comptime] value_config: VM::Config) -> Self {
        let key = SM::allocate_rhs(score_config);
        let value = VM::allocate_rhs(value_config);
        SeparateKV::<AP, SM, VM> { key, value }
    }
}

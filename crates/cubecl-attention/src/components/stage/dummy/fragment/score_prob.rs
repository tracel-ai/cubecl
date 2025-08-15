use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;
use crate::components::tile::{ScoreMatmul, ValueMatmul};

#[derive(CubeType)]
pub enum ScoreProbFragment<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    Reuse(ReuseSP<AP, SM, VM>),
    Separate(SeparateSP<AP, SM, VM>),
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>>
    ScoreProbFragment<AP, SM, VM>
{
    pub fn new<S: StageAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>>(
        #[comptime] config: S,
    ) -> Self {
        match config.reuse_key_value() {
            true => Self::new_Reuse(ReuseSP::new(config.score_config(), config.value_config())),
            false => Self::new_Separate(SeparateSP::new(
                config.score_config(),
                config.value_config(),
            )),
        }
    }

    pub fn score(&self) -> &SM::Accumulator {
        match self {
            ScoreProbFragment::Reuse(reuse_sp) => &reuse_sp.fragment,
            ScoreProbFragment::Separate(separate_sp) => &separate_sp.score,
        }
    }

    pub fn score_mut(&mut self) -> &mut SM::Accumulator {
        match self {
            ScoreProbFragment::Reuse(reuse_sp) => &mut reuse_sp.fragment,
            ScoreProbFragment::Separate(separate_sp) => &mut separate_sp.score,
        }
    }

    pub fn prob(&self) -> &VM::Lhs {
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => &separate_sp.prob,
        }
    }

    pub fn prob_mut(&mut self) -> &mut VM::Lhs {
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => &mut separate_sp.prob,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseSP<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    pub fragment: SM::Accumulator,
    #[cube(comptime)]
    _phantom: PhantomData<VM>,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> ReuseSP<AP, SM, VM> {
    pub fn new(
        #[comptime] score_config: SM::Config,
        #[comptime] _value_config: VM::Config,
    ) -> Self {
        let fragment = SM::allocate_accumulator(score_config);
        ReuseSP::<AP, SM, VM> {
            fragment,
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType)]
pub struct SeparateSP<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    pub score: SM::Accumulator,
    pub prob: VM::Lhs,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> SeparateSP<AP, SM, VM> {
    pub fn new(#[comptime] score_config: SM::Config, #[comptime] value_config: VM::Config) -> Self {
        let score = SM::allocate_accumulator(score_config);
        let prob = VM::allocate_lhs(value_config);
        SeparateSP::<AP, SM, VM> { score, prob }
    }
}

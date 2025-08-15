use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub struct KeyValue {}
#[derive(CubeType)]
pub struct ScoreProb<AP: AttentionPrecision> {
    #[cube(comptime)]
    _phantom: PhantomData<AP>,
}

#[cube]
impl<AP: AttentionPrecision> ScoreProb<AP> {
    pub fn new<S: StageAttentionConfig>(
        #[comptime] _score_config: S::ScoreConfig,
        #[comptime] _value_config: S::ValueConfig,
    ) -> ScoreProb<AP> {
        ScoreProb::<AP> {
            _phantom: PhantomData,
        }
    }
}

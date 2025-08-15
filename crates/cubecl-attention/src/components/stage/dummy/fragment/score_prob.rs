use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;
use crate::components::tile::ScoreProb;

#[derive(CubeType)]
pub enum ScoreProbFragment<AP: AttentionPrecision> {
    Reuse(ScoreProb<AP>),
    Separate(ScoreProb<AP>, ScoreProb<AP>),
}

#[cube]
impl<AP: AttentionPrecision> ScoreProbFragment<AP> {
    pub fn new<S: StageAttentionConfig>(#[comptime] config: S) -> Self {
        match config.reuse_key_value() {
            true => Self::new_Reuse(ScoreProb::new::<S>(
                config.score_config(),
                config.value_config(),
            )),
            false => Self::new_Separate(
                ScoreProb::new::<S>(config.score_config(), config.value_config()),
                ScoreProb::new::<S>(config.score_config(), config.value_config()),
            ),
        }
    }

    pub fn score(&self) -> &ScoreProb<AP> {
        match self {
            ScoreProbFragment::Reuse(score_prob) => score_prob,
            ScoreProbFragment::Separate(score, _) => score,
        }
    }

    pub fn score_mut(&mut self) -> &mut ScoreProb<AP> {
        match self {
            ScoreProbFragment::Reuse(score_prob) => score_prob,
            ScoreProbFragment::Separate(score, _) => score,
        }
    }

    pub fn prob(&self) -> &ScoreProb<AP> {
        match self {
            ScoreProbFragment::Reuse(score_prob) => score_prob,
            ScoreProbFragment::Separate(_, prob) => prob,
        }
    }

    pub fn prob_mut(&mut self) -> &mut ScoreProb<AP> {
        match self {
            ScoreProbFragment::Reuse(score_prob) => score_prob,
            ScoreProbFragment::Separate(_, prob) => prob,
        }
    }
}

use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::AttentionPrecision;
use crate::components::tile::{ScoreMatmul, TileAttentionConfig, ValueMatmul};

#[derive(CubeType)]
pub enum ScoreProbFragment<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> {
    Reuse(ReuseSP<AP, SM, VM>),
    Separate(SeparateSP<AP, SM, VM>),
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>>
    ScoreProbFragment<AP, SM, VM>
{
    pub fn new<T: TileAttentionConfig<ScoreConfig = SM::Config, ValueConfig = VM::Config>>(
        #[comptime] config: T,
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

    pub fn multiply_score(&mut self, factor: AP::EA) {
        comment!("Multiply score");
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => separate_sp.multiply_score(factor),
        }
    }
    pub fn row_max(&mut self, base: AP::EA) -> AP::EA {
        comment!("Row Max");
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => separate_sp.row_max(base),
        }
    }
    pub fn to_prob(&mut self, m: AP::EA) {
        comment!("To Prob");
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => separate_sp.to_prob(m),
        }
    }

    pub fn row_sum(&self) -> AP::EA {
        comment!("Row sum");
        match self {
            ScoreProbFragment::Reuse(_reuse_sp) => comptime!(todo!()),
            ScoreProbFragment::Separate(separate_sp) => separate_sp.row_sum(),
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
    tmp_smem: SharedMemory<AP::EA>,
    pub score: SM::Accumulator,
    pub prob: VM::Lhs,
    #[cube(comptime)]
    score_config: SM::Config,
    #[cube(comptime)]
    value_config: VM::Config,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>, VM: ValueMatmul<AP>> SeparateSP<AP, SM, VM> {
    pub fn new(#[comptime] score_config: SM::Config, #[comptime] value_config: VM::Config) -> Self {
        let mut score = SM::allocate_accumulator(score_config);
        SM::zero_accumulator(&mut score, score_config);
        let prob = VM::allocate_lhs(value_config);
        SeparateSP::<AP, SM, VM> {
            tmp_smem: SharedMemory::<AP::EA>::new(64),
            score,
            prob,
            score_config,
            value_config,
        }
    }

    pub fn multiply_score(&mut self, factor: AP::EA) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        SM::write_results::<AP::EA>(
            &self.score,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.score_config,
        );
        self.tmp_smem[index_0] *= factor;
        self.tmp_smem[index_1] *= factor;
        sync_plane();
    }

    pub fn row_max(&mut self, base: AP::EA) -> AP::EA {
        let row = UNIT_POS_X / 4;
        let row_offset = row * 8;
        let mut rowmax = base;

        for i in 0..8 {
            let ts = self.tmp_smem[row_offset + i];
            if ts > rowmax {
                rowmax = ts;
            }
        }

        rowmax
    }

    pub fn to_prob(&mut self, m: AP::EA) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        self.tmp_smem[index_0] = Exp::exp(self.tmp_smem[index_0] - m);
        self.tmp_smem[index_1] = Exp::exp(self.tmp_smem[index_1] - m);
        sync_plane();

        let prob_tile = Tile::<AP::EA> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };

        VM::fill_lhs(&prob_tile, &mut self.prob, self.value_config);
    }

    // TODO: Must be done after to_prob (encode that in type system)
    pub fn row_sum(&self) -> AP::EA {
        let row = UNIT_POS_X / 4;
        let row_offset = row * 8;

        let mut rowsum = AP::EA::from_int(0);
        for i in 0..8 {
            rowsum += self.tmp_smem[row_offset + i];
        }

        rowsum
    }
}

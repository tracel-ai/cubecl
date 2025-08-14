use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulPrecision;
use cubecl_matmul::components::tile::TileMatmul;

use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub enum ScoreProbFragment<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> {
    Reuse(ReuseSP<MP, STM, VTM>),
    Separate(SeparateSP<MP, STM, VTM>),
}

#[cube]
impl<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> ScoreProbFragment<MP, STM, VTM>
{
    pub fn new<S: StageAttentionConfig<ScoreConfig = STM::Config, ValueConfig = VTM::Config>>(
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

    pub fn score(&self) -> &STM::Accumulator {
        match self {
            ScoreProbFragment::Reuse(reuse_sp) => reuse_sp.fragment.score(),
            ScoreProbFragment::Separate(separate_sp) => &separate_sp.score,
        }
    }

    pub fn score_mut(&mut self) -> &mut STM::Accumulator {
        match self {
            ScoreProbFragment::Reuse(reuse_sp) => reuse_sp.fragment.score_mut(),
            ScoreProbFragment::Separate(separate_sp) => &mut separate_sp.score,
        }
    }

    pub fn to_prob(self) -> VTM::Lhs {
        match self {
            ScoreProbFragment::Reuse(reuse_sp) => reuse_sp.fragment.to_prob(),
            ScoreProbFragment::Separate(separate_sp) => separate_sp.prob,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseSP<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> {
    // Will be cast to VTM
    pub fragment: ReusedFragment<MP, STM, VTM>,
    #[cube(comptime)]
    _phantom: PhantomData<VTM>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> ReuseSP<MP, STM, VTM>
{
    pub fn new(
        #[comptime] score_config: STM::Config,
        #[comptime] value_config: VTM::Config,
    ) -> Self {
        ReuseSP::<MP, STM, VTM> {
            fragment: ReusedFragment::new(score_config, value_config),
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType)]
pub struct SeparateSP<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> {
    pub score: STM::Accumulator,
    pub prob: VTM::Lhs,
}

#[cube]
impl<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> SeparateSP<MP, STM, VTM> {
    pub fn new(
        #[comptime] score_config: STM::Config,
        #[comptime] value_config: VTM::Config,
    ) -> Self {
        let score = STM::allocate_accumulator(score_config);
        let prob = VTM::allocate_lhs(value_config);
        SeparateSP::<MP, STM, VTM> { score, prob }
    }
}

#[derive(CubeType)]
pub struct ReusedFragment<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> {
    pub frag: STM::Accumulator,
    #[cube(comptime)]
    _phantom: PhantomData<VTM>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Lhs: FromAccumulator<STM::Accumulator>>,
> ReusedFragment<MP, STM, VTM>
{
    pub fn new(
        #[comptime] score_config: STM::Config,
        #[comptime] _value_config: VTM::Config,
    ) -> Self {
        ReusedFragment::<MP, STM, VTM> {
            frag: STM::allocate_accumulator(score_config),
            _phantom: PhantomData,
        }
    }

    pub fn to_prob(self) -> VTM::Lhs {
        <VTM::Lhs as FromAccumulator<STM::Accumulator>>::from_accumulator(self.frag)
    }

    pub fn score(&self) -> &STM::Accumulator {
        &self.frag
    }

    pub fn score_mut(&mut self) -> &mut STM::Accumulator {
        &mut self.frag
    }
}

#[cube]
pub trait FromAccumulator<Acc: CubeType>: CubeType {
    fn from_accumulator(acc: Acc) -> Self;
}

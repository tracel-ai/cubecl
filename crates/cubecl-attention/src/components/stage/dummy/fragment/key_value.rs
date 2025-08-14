use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulPrecision;
use cubecl_matmul::components::tile::TileMatmul;

use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub enum KeyValueFragment<
    MP: MatmulPrecision,
    STM: TileMatmul<MP>,
    VTM: TileMatmul<MP, Rhs = STM::Rhs>,
> {
    Reuse(ReuseKV<MP, STM, VTM>),
    Separate(SeparateKV<MP, STM, VTM>),
}

#[cube]
impl<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP, Rhs = STM::Rhs>>
    KeyValueFragment<MP, STM, VTM>
{
    pub fn new<S: StageAttentionConfig<ScoreConfig = STM::Config, ValueConfig = VTM::Config>>(
        #[comptime] config: S,
    ) -> Self {
        match config.reuse_key_value() {
            true => Self::new_Reuse(ReuseKV::new(config.score_config(), config.value_config())),
            false => Self::new_Separate(SeparateKV::new(
                config.score_config(),
                config.value_config(),
            )),
        }
    }

    pub fn key(&self) -> &STM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &separate_kv.key,
        }
    }

    pub fn key_mut(&mut self) -> &mut STM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.key,
        }
    }

    pub fn value(&self) -> &VTM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &separate_kv.value,
        }
    }

    pub fn value_mut(&mut self) -> &mut VTM::Rhs {
        match self {
            KeyValueFragment::Reuse(reuse_kv) => &mut reuse_kv.fragment,
            KeyValueFragment::Separate(separate_kv) => &mut separate_kv.value,
        }
    }
}

#[derive(CubeType)]
pub struct ReuseKV<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> {
    // Will be cast to VTM
    pub fragment: STM::Rhs,
    #[cube(comptime)]
    _phantom: PhantomData<VTM>,
}

#[cube]
impl<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> ReuseKV<MP, STM, VTM> {
    pub fn new(
        #[comptime] score_config: STM::Config,
        #[comptime] _value_config: VTM::Config,
    ) -> Self {
        let fragment = STM::allocate_rhs(score_config);
        ReuseKV::<MP, STM, VTM> {
            fragment,
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType)]
pub struct SeparateKV<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> {
    pub key: STM::Rhs,
    pub value: VTM::Rhs,
}

#[cube]
impl<MP: MatmulPrecision, STM: TileMatmul<MP>, VTM: TileMatmul<MP>> SeparateKV<MP, STM, VTM> {
    pub fn new(
        #[comptime] score_config: STM::Config,
        #[comptime] value_config: VTM::Config,
    ) -> Self {
        let key = STM::allocate_rhs(score_config);
        let value = VTM::allocate_rhs(value_config);
        SeparateKV::<MP, STM, VTM> { key, value }
    }
}

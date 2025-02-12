use crate::matmul::components::global::multi_stage::buffer_loading::BufferLoading;
use crate::matmul::components::global::{
    CommonGlobalConfig, GlobalConfig, GlobalMatmulFamily, LoadingValidation,
};
use crate::matmul::components::stage::single_buffer::{
    LhsBufferReaderFamily, RhsBufferReaderFamily,
};
use crate::matmul::components::stage::StageConfig;
use crate::matmul::components::tile::TileConfig;
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{stage, MatmulPrecision};
use crate::matmul::components::{Ident, InvalidConfigError};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::check_buffers_contiguous;
use super::DoubleBufferingMatmul;

pub struct DoubleBufferingMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<
        LhsReader = LhsBufferReaderFamily,
        RhsReader = RhsBufferReaderFamily,
    >,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingMatmul<MP, SMM::Matmul<MP::ES, MP::EG, MP::EA>>;
}

impl<SMM> MatmulConfigFactory for DoubleBufferingMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Input = SMM::Input;
    type Config = CommonGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        check_buffers_contiguous::<Self::Config>(Ident::Lhs, config)?;
        check_buffers_contiguous::<Self::Config>(Ident::Rhs, config)?;

        let tmm_config = config.smm_config.to_tmm_config();
        let tile_shape = tmm_config.tile_shape();

        if tile_shape.m != tile_shape.n || tile_shape.n != tile_shape.k {
            return Err(Box::new("Only support square tiling"));
        }

        BufferLoading::check::<Self::Config>(config, Ident::Lhs)?;
        BufferLoading::check::<Self::Config>(config, Ident::Rhs)?;

        if config.stage_tiling(Ident::Lhs).tile_count_col() != 2 {
            return Err(Box::new("Pipelined matmul needs exactly 2 buffers."));
        }

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.smm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let smm_config = SMM::make_config(input, problem, cube_dim, cube_count, advanced_config);
        let stage_shape = SMM::stage_shape(&smm_config);

        CommonGlobalConfig::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % stage_shape.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
        )
    }
}

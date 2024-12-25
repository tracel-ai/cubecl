use crate::matmul::components::global::buffered::buffer_loading::BufferLoading;
use crate::matmul::components::global::{
    CommonGlobalConfig, GlobalConfig, GlobalMatmulFamily, LoadingValidation,
};
use crate::matmul::components::stage::single_buffer::{
    LhsBufferReaderFamily, RhsBufferReaderFamily,
};
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{stage, MatmulPrecision};
use crate::matmul::components::{Ident, InvalidConfigError};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::check_buffers_contiguous;
use super::PipelinedMatmul;

pub struct PipelinedMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for PipelinedMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<
        LhsReader = LhsBufferReaderFamily,
        RhsReader = RhsBufferReaderFamily,
    >,
{
    type Matmul<MP: MatmulPrecision> = PipelinedMatmul<MP, SMM::Matmul<MP::ES, MP::EG, MP::EA>>;
}

impl<SMM> MatmulConfigFactory for PipelinedMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Input = SMM::Input;
    type Config = CommonGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        check_buffers_contiguous::<Self::Config>(Ident::Lhs, &config)?;
        check_buffers_contiguous::<Self::Config>(Ident::Rhs, &config)?;

        BufferLoading::check::<Self::Config>(&config, Ident::Lhs)?;
        BufferLoading::check::<Self::Config>(&config, Ident::Rhs)?;

        if config.stage_dim(Ident::Lhs).num_tiles_y_dim() != 2 {
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
        let size = SMM::size(&smm_config);

        CommonGlobalConfig::new(
            smm_config,
            problem.m as u32 % size.m != 0,
            problem.n as u32 % size.n != 0,
            problem.k as u32 % size.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
        )
    }
}

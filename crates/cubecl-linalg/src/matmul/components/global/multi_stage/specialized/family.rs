use std::marker::PhantomData;

use cubecl_core::{client::ComputeClient, CubeCount, CubeDim, Runtime};

use crate::matmul::{
    components::{
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{
            self,
            single_buffer::{LhsBufferReaderFamily, RhsBufferReaderFamily},
        },
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem,
    },
    kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
};

use super::{config::Config, SpecializedMatmul};

pub struct SpecializedMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for SpecializedMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<
        LhsReader = LhsBufferReaderFamily,
        RhsReader = RhsBufferReaderFamily,
    >,
{
    type Matmul<MP: MatmulPrecision> = SpecializedMatmul<MP, SMM::Matmul<MP::ES, MP::EG, MP::EA>>;
}

impl<SMM> MatmulConfigFactory for SpecializedMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Input = SMM::Input;
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        if config.num_producers() == 0 {
            return Err(Box::new("There are no producer planes. Make sure there are more planes than the underlying stage matmul requires."));
        }
        if config.stage_tiling(Ident::Lhs).tile_count_col() <= 1 {
            return Err(Box::new("Producer-consumer needs at least 2 buffers."));
        }

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
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

        Config::new(
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

use std::marker::PhantomData;

use cubecl_core::{client::ComputeClient, CubeCount, CubeDim, Runtime};

use crate::matmul::{
    components::{
        global::{
            single_stage::{loader::LoadingStrategy, Config},
            GlobalConfig, GlobalMatmulFamily,
        },
        stage::{
            self,
            multi_buffer::{LhsReaderFamily, RhsReaderFamily},
        },
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem,
    },
    kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
};

use super::SimplePipelinedMatmul;

pub struct SimpleMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: LoadingStrategy,
    RL: LoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = LhsReaderFamily, RhsReader = RhsReaderFamily>,
    LL: LoadingStrategy,
    RL: LoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SimplePipelinedMatmul<MP, SMM::Matmul<MP::ES, MP::EG, MP::EA>, LL, RL>;
}

impl<SMM, LL, RL> MatmulConfigFactory for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: LoadingStrategy,
    RL: LoadingStrategy,
{
    type Input = SMM::Input;
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check(config, Ident::Lhs)?;
        RL::check(config, Ident::Rhs)?;
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
            stage_shape.k,
        )
    }
}

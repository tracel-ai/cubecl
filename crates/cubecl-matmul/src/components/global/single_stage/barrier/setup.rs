use std::marker::PhantomData;

use crate::components::LoadSpecializationConfig;
use crate::components::MatmulChecker;
use crate::components::MatmulPrecision;
use crate::components::global::load::AsyncFullLoadingStrategy;
use crate::components::global::single_stage::SingleStageConfig;
use crate::components::global::single_stage::barrier::matmul::SimpleBarrierMatmul;
use crate::components::problem::MatmulLineSizes;
use crate::components::stage::FullReaderFamily;
use crate::components::stage::StageConfig;
use crate::kernels::matmul::GlobalInput;
use crate::kernels::matmul::MatmulSelection;
use crate::{
    components::{
        Ident, InvalidConfigError, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage,
    },
    kernels::MatmulAvailabilityError,
};
use cubecl_core::Feature;
use cubecl_core::{CubeDim, Runtime, client::ComputeClient};

pub struct SimpleBarrierMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SimpleBarrierMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
    type Input = GlobalInput<SMM::Input>;

    fn cube_dim(
        selection: &MatmulSelection,
        load_specialization: LoadSpecializationConfig,
    ) -> Result<CubeDim, InvalidConfigError> {
        let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
            .as_plane_resources(selection.plane_dim)?
            .get_count();

        if let LoadSpecializationConfig::None = load_specialization {
            Ok(CubeDim::new_2d(selection.plane_dim, main_flow_planes))
        } else {
            Err(Box::new(
                "Error: Specialization is unavailable for simple barrier matmul.",
            ))
        }
    }

    fn setup(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        quantized: bool,
    ) -> Self::Config {
        let stage_config = SMM::setup(input.stage_input, problem, line_sizes, cube_dim, quantized);
        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        SingleStageConfig::new(
            stage_config,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % stage_shape_k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            line_sizes.lhs as u32,
            line_sizes.rhs as u32,
            line_sizes.out as u32,
            stage_shape_k,
            input.loading_precompute_strategy,
            input.loader_mode,
        )
    }
}

impl<SMM, LL, RL> MatmulChecker for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Config = SingleStageConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check(config, Ident::Lhs)?;
        RL::check(config, Ident::Rhs)?;
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config())?;

        if !client.properties().feature_enabled(Feature::Barrier) {
            return Err(MatmulAvailabilityError::BarrierUnavailable);
        }

        Ok(())
    }
}

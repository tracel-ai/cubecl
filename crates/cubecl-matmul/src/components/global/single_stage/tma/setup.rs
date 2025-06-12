use crate::components::LoadSpecializationConfig;
use crate::components::MatmulChecker;
use crate::components::MatmulPrecision;
use crate::components::global::GlobalConfig as _;
use crate::components::global::load::TmaTiling;
use crate::components::global::single_stage::SingleStageConfig;
use crate::components::global::single_stage::tma::matmul::SimpleTmaMatmul;
use crate::components::problem::MatmulLineSizes;
use crate::components::stage::StageConfig;
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::{GlobalInput, MatmulSelection};
use std::any::TypeId;
use std::marker::PhantomData;

use cubecl_core::CubeDim;
use cubecl_core::Feature;
use cubecl_core::Runtime;
use cubecl_core::TmaFeature;
use cubecl_core::client::ComputeClient;
use cubecl_core::tf32;

use crate::components::{
    InvalidConfigError, MatmulProblem,
    global::GlobalMatmulFamily,
    stage::{self, FullReaderFamily},
};

pub struct SimpleTmaMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for SimpleTmaMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
{
    type Matmul<MP: MatmulPrecision> = SimpleTmaMatmul<MP, SMM::Matmul<MP, TmaTiling, TmaTiling>>;
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
                "Error: Specialization is unavailable for simple tma matmul.",
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
        let mut line_sizes = line_sizes.clone();

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        line_sizes.lhs = 1;
        line_sizes.rhs = 1;

        let stage_config = SMM::setup(input.stage_input, problem, &line_sizes, cube_dim, quantized);
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

impl<SMM> MatmulChecker for SimpleTmaMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Config = SingleStageConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config())?;

        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        if !client
            .properties()
            .feature_enabled(Feature::Tma(TmaFeature::Base))
        {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        Ok(())
    }
}

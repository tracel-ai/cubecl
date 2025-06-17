use crate::components::AvailableLineSizes;
use crate::components::LoadSpecializationConfig;
use crate::components::MatmulPrecision;
use crate::components::global::load::NoLoadingValidation;
use crate::components::global::load::TmaTiling;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::global::single_stage::tma::matmul::SimpleTmaMatmul;
use crate::components::stage::StageConfig;
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;

use crate::components::{
    MatmulProblem,
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
    type Config = SimpleTmaConfig<SMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        mut available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        available_line_sizes.lhs = vec![1];
        available_line_sizes.rhs = vec![1];

        let stage_config = SMM::setup::<MP, R>(
            client,
            problem,
            selection,
            available_line_sizes,
            (1, 1).into(),
        )?;
        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        let num_planes =
            if let LoadSpecializationConfig::None = selection.load_specialization_config {
                stage_config.num_main_flow_planes()
            } else {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Error: Specialization is unavailable for simple tma matmul.",
                )));
            };

        SimpleTmaConfig::new::<NoLoadingValidation, NoLoadingValidation, MP, R>(
            client,
            stage_config,
            num_planes,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % stage_shape_k != 0,
            stage_shape_k,
            selection.loading_precompute_strategy,
            selection.loader_mode,
        )
    }
}

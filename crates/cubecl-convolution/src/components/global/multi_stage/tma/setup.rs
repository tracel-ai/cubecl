use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{
    AvailableLineSizes, MatmulLineSizes, MatmulPrecision, MatmulSelection, MatmulSetupError,
    global::{read::NoLoadingValidation, single_stage::tma::SimpleTmaConfig},
    stage::{FullStageReaderFamily, StageConfig as _, StageMatmulFamily},
};
use cubecl_std::tensor::layout::Coords2d;

use crate::{
    components::{
        ConvolutionConfig, ConvolutionProblem,
        global::{
            GlobalConvolutionFamily,
            multi_stage::tma::{MultiStageTmaConvolution, num_stages},
            read::{im2col_tma::TmaIm2colTiling, weight_tma::TmaWeightTiling},
        },
        stage::reader::BiasTilingLayout,
    },
    kernels::layered::algorithm::simple_tma::check_problem_tma,
};

pub struct MultiStageTmaConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

impl<SMM> GlobalConvolutionFamily for MultiStageTmaConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<
            LhsStageReader = FullStageReaderFamily,
            RhsStageReader = FullStageReaderFamily,
            AccStageReader = Option<FullStageReaderFamily>,
            WriteCoords = Coords2d,
        >,
{
    type Convolution<MP: MatmulPrecision> = MultiStageTmaConvolution<
        MP,
        SMM::Matmul<MP, TmaIm2colTiling, TmaWeightTiling, BiasTilingLayout>,
    >;
    type Config = ConvolutionConfig<SimpleTmaConfig<SMM::Config>>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
    }

    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        check_problem_tma(problem)?;

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        assert!(line_sizes.lhs == 1);
        assert!(line_sizes.rhs == 1);

        let stage_config = SMM::setup::<MP, R>(
            client,
            &problem.as_matmul_problem(),
            selection,
            line_sizes,
            // Not the same as num_stages
            (1, 1).into(),
            None,
            false,
        )?;

        let stage_k = stage_config.tiling_scheme().elements_in_stage_k();

        let num_stages = num_stages::<R, MP>(
            client,
            problem,
            stage_config.num_main_flow_planes(),
            &stage_config.tiling_scheme(),
        );

        ConvolutionConfig::new(
            SimpleTmaConfig::new::<NoLoadingValidation, NoLoadingValidation, MP, R>(
                client,
                stage_config,
                stage_config.num_main_flow_planes(),
                // TODO: Find the correct condition to avoid check bounds.
                true,
                true,
                true,
                stage_k,
                selection.loading_precompute_strategy,
                selection.reader_mode,
            )?,
            &problem.kernel_size,
            &problem.stride,
            &problem.dilation,
            &problem.padding,
            problem.dimensionality,
            num_stages,
        )
    }
}

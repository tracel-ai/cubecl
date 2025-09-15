use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{
    AvailableLineSizes, MatmulLineSizes, MatmulPrecision, MatmulSelection, MatmulSetupError,
    global::{load::NoLoadingValidation, single_stage::simple::SimpleConfig},
    stage::{
        ContiguousTilingLayout, FullReaderFamily, RowMajorTilingOrder, StageConfig as _,
        StageMatmulFamily,
    },
};
use cubecl_std::tensor::layout::Coords3d;

use crate::components::{
    ConvolutionConfig, ConvolutionProblem,
    global::{GlobalConvolutionFamily, single_stage::simple::SimpleConvolution},
    stage::reader::BiasTilingLayout,
};

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

pub struct SimpleConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

impl<SMM> GlobalConvolutionFamily for SimpleConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<
            LhsReader = FullReaderFamily,
            RhsReader = FullReaderFamily,
            AccReader = Option<FullReaderFamily>,
            WriteCoords = Coords3d,
        >,
{
    type Convolution<MP: MatmulPrecision> = SimpleConvolution<
        MP,
        SMM::Matmul<MP, ConvTilingLayout, ConvTilingLayout, BiasTilingLayout>,
    >;
    type Config = ConvolutionConfig<SimpleConfig<SMM::Config>>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let stage_config = SMM::setup::<MP, R>(
            client,
            &problem.as_matmul_problem(),
            selection,
            line_sizes,
            (1, 1).into(),
            None,
            false,
        )?;
        let stage_k = stage_config.tiling_scheme().elements_in_stage_k();

        ConvolutionConfig::new(
            SimpleConfig::new::<NoLoadingValidation, NoLoadingValidation, MP, R>(
                client,
                stage_config,
                stage_config.num_main_flow_planes(),
                true,
                true,
                true,
                stage_k,
                selection.loading_precompute_strategy,
                selection.loader_mode,
            )?,
            &problem.kernel_size,
            &problem.stride,
            &problem.dilation,
            &problem.padding,
            problem.dimensionality,
            1,
        )
    }
}

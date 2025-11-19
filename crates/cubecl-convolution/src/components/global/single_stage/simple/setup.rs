use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{
    AvailableLineSizes, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulSelection,
    MatmulSetupError,
    global::{PartitionedStageFamily, WriteTiling, read::NoLoadingValidation},
    stage::{
        ContiguousTilingLayout, RowMajorTilingOrder, StageConfig as _, StageMatmulFamily,
        StridedStageFamily, TilingLayout, TilingLayoutConfig, TilingLayoutEnum,
    },
};

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
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
{
    type Convolution<MP: MatmulPrecision> = SimpleConvolution<
        MP,
        SMM::Matmul<MP, ConvTilingLayout, ConvTilingLayout, BiasTilingLayout, WriteTiling>,
    >;
    type Config = ConvolutionConfig<SimpleConfig<SMM::Config>>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tiling_layout = TilingLayoutConfig {
            lhs: ConvTilingLayout::to_enum(),
            rhs: ConvTilingLayout::to_enum(),
            acc: TilingLayoutEnum::Other,
            out: WriteTiling::to_enum(),
        };
        let stage_config = SMM::setup::<R>(
            client,
            &problem.as_matmul_problem(),
            selection,
            line_sizes,
            tiling_layout,
            (1, 1).into(),
            None,
            false,
            dtypes,
        )?;
        let stage_k = stage_config.tiling_scheme().elements_in_stage_k();

        ConvolutionConfig::new(
            SimpleConfig::new::<NoLoadingValidation, NoLoadingValidation, R>(
                client,
                stage_config,
                stage_config.num_main_flow_planes(),
                true,
                true,
                true,
                stage_k,
                selection.loading_precompute_strategy,
                selection.reader_mode,
                dtypes,
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

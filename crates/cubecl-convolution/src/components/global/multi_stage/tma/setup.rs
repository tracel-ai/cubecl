use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient};
use cubecl_matmul::components::{
    AvailableLineSizes, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulSelection,
    MatmulSetupError, MatrixLayout, StageIdent,
    global::{
        GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts, PartitionedStageFamily,
        SharedGlobalMatmulConfig, WriteTiling, cube_dim_validation,
        memory::{GlobalMemoryConfig, ViewDirection},
        multi_stage::EventLoadingMode,
        read::{validate_async_barrier, validate_tma},
    },
    stage::{StageConfig as _, StageMatmulFamily, StridedStageFamily},
};

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
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = Option<StridedStageFamily>,
            OutStage = PartitionedStageFamily,
        >,
{
    type Convolution<MP: MatmulPrecision> = MultiStageTmaConvolution<
        MP,
        SMM::Matmul<MP, TmaIm2colTiling, TmaWeightTiling, BiasTilingLayout, WriteTiling>,
    >;
    type Config = ConvolutionConfig<SharedGlobalMatmulConfig<SMM::Config>>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        check_problem_tma(problem)?;
        validate_async_barrier(client)?;

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        assert!(line_sizes.lhs == 1);
        assert!(line_sizes.rhs == 1);

        let stage_config = SMM::setup(
            client,
            &problem.as_matmul_problem(),
            selection,
            line_sizes,
            // Not the same as num_stages
            (1, 1).into(),
            None,
            dtypes,
        )?;

        let check_m_bounds = true;
        let check_n_bounds = true;
        let check_k_bounds = true;

        let plane_role_config = stage_config.plane_role_config();
        let plane_counts = MatmulPlaneCounts::new(
            selection.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let num_stages = num_stages(
            client,
            problem,
            stage_config.num_main_flow_planes(),
            &selection.tiling_scheme,
            dtypes,
        );
        let precompute_job = selection.loading_precompute_strategy.into();
        let plane_dim = selection.plane_dim;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let reader_mode = selection.reader_mode;

        let lhs_smem_config = stage_config.lhs_smem_config();
        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_k_bounds,
            matrix_layout: problem.lhs_layout,
            view_direction: ViewDirection::Col,
        };

        let rhs_smem_config = stage_config.rhs_smem_config();
        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs as u32,
            check_row_bounds: check_k_bounds,
            check_col_bounds: check_n_bounds,
            matrix_layout: problem.rhs_layout,
            view_direction: ViewDirection::Row,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_n_bounds,
            view_direction: ViewDirection::None,
        };

        let out_smem_config = stage_config.out_smem_config();

        let lhs_reader_config = GlobalReaderConfig {
            gmem_config: lhs_gmem_config,
            smem_config: lhs_smem_config,
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Lhs,
            event_loading_mode,
            specialization_tensor_config: selection.load_specialization_config.lhs,
        };

        let rhs_reader_config = GlobalReaderConfig {
            gmem_config: rhs_gmem_config,
            smem_config: rhs_smem_config,
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Rhs,
            event_loading_mode,
            specialization_tensor_config: selection.load_specialization_config.rhs,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: out_smem_config,
            role_rule_config: plane_role_config.rule,
            plane_dim: selection.plane_dim,
        };

        validate_tma(
            client,
            &problem.as_matmul_problem(),
            &lhs_reader_config,
            dtypes,
        )?;
        validate_tma(
            client,
            &problem.as_matmul_problem(),
            &rhs_reader_config,
            dtypes,
        )?;

        let matmul_config = SharedGlobalMatmulConfig {
            stage_config,
            num_planes: plane_counts.total,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
            must_sync_plane_after_execution: false,
        };

        cube_dim_validation(matmul_config)?;

        ConvolutionConfig::new(
            matmul_config,
            &problem.kernel_size,
            &problem.stride,
            &problem.dilation,
            &problem.padding,
            problem.dimensionality,
            num_stages,
        )
    }
}

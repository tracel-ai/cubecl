use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{
    LoadingPrecomputeStrategy, MatrixLayout, StageIdent,
    global::{
        GlobalReaderConfig, GlobalWriterConfig, PartitionedStageFamily, PlaneRoleConfig,
        RoleRuleConfig, SpecializationTensorConfig,
        memory::{GlobalMemoryConfig, ViewDirection},
        multi_stage::EventLoadingMode,
        read::ReaderMode,
    },
    stage::StridedStageFamily,
};

use crate::components::{
    AttentionElems, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    global::{
        GlobalAttentionFamily,
        simple::{SimpleGlobalAttention, config::SimpleGlobalAttentionConfig},
    },
    stage::{StageAttentionConfig as _, StageAttentionFamily},
};

pub struct SimpleGlobalAttentionFamily<SA: StageAttentionFamily> {
    _phantom: PhantomData<SA>,
}

impl<
    SA: StageAttentionFamily<
            KeyStage = StridedStageFamily,
            ValueStage = StridedStageFamily,
            OutStage = PartitionedStageFamily,
        >,
> GlobalAttentionFamily for SimpleGlobalAttentionFamily<SA>
{
    type Attention<AP: AttentionPrecision> = SimpleGlobalAttention<AP, SA::Attention<AP>>;

    type Config = SimpleGlobalAttentionConfig<SA::Config>;

    fn setup<R: cubecl_core::Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let stage_config = SA::setup(client, problem, selection, line_sizes, dtypes)?;

        let precompute_job = LoadingPrecomputeStrategy::Never.into();
        let plane_dim = stage_config.plane_dim();
        let reader_mode = ReaderMode::Relaxed;
        let event_loading_mode = EventLoadingMode::Relaxed;
        let specialization_tensor_config = SpecializationTensorConfig::MainFlowOnly;
        let plane_role_config = PlaneRoleConfig::new_unspecialized(stage_config.num_planes());

        let query_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.query as u32,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
            view_direction: ViewDirection::None,
        };

        let mask_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.mask as u32,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
            view_direction: ViewDirection::Col,
        };

        let key_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.key as u32,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
            view_direction: ViewDirection::Row,
        };

        let value_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.value as u32,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
            view_direction: ViewDirection::Row,
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
            view_direction: ViewDirection::None,
        };

        let key_reader_config = GlobalReaderConfig {
            gmem_config: key_gmem_config,
            smem_config: stage_config.key_smem_config(),
            precompute_job,
            plane_dim,
            reader_mode,
            event_loading_mode,
            specialization_tensor_config,
            plane_role_config,
            stage_ident: StageIdent::Rhs,
        };

        let value_reader_config = GlobalReaderConfig {
            gmem_config: value_gmem_config,
            smem_config: stage_config.value_smem_config(),
            precompute_job,
            plane_dim,
            reader_mode,
            event_loading_mode,
            specialization_tensor_config,
            plane_role_config,
            stage_ident: StageIdent::Rhs,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: stage_config.out_smem_config(),
            role_rule_config: RoleRuleConfig::MainFlowOnly,
            plane_dim,
        };

        Ok(SimpleGlobalAttentionConfig {
            stage_config,
            key_reader_config,
            value_reader_config,
            query_gmem_config,
            mask_gmem_config,
            writer_config,
        })
    }
}

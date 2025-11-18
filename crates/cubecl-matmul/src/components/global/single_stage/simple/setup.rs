use crate::components::{
    MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulSelection, MatrixLayout, StageIdent,
    error::MatmulSetupError,
    global::{
        GlobalConfig as _, GlobalReaderConfig, GlobalWriterConfig, GlobalWriterFamily,
        SharedGlobalConfig, SpecializationTensorConfig, WriteTiling, cube_dim_validation,
        memory::{GlobalMemoryConfig, ViewDirection},
        multi_stage::EventLoadingMode,
        read::{FullLoadingStrategy, LoadingValidation},
        single_stage::simple::matmul::SimpleMatmul,
    },
    stage::{
        FilledStageFamily, NoTilingLayout, StageConfig, StageMemoryConfig, StridedStageFamily,
    },
};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    global::GlobalMatmulFamily,
    stage::{self},
};

/// Simple matmul family for any precision
pub struct SimpleMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = SimpleMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout, WriteTiling>,
        LL,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalConfig<SMM::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        // let tiling_layout = TilingLayoutConfig {
        //     lhs: LL::TilingLayout::to_enum(),
        //     rhs: RL::TilingLayout::to_enum(),
        //     acc: TilingLayoutEnum::Other,
        //     out: WriteTiling::to_enum(),
        // };
        let stage_config = SMM::setup::<R>(
            client,
            problem,
            selection,
            line_sizes,
            (1, 1).into(),
            None,
            false,
            dtypes,
        )?;

        let stage_shape_m = stage_config.elements_in_stage_m();
        let stage_shape_n = stage_config.elements_in_stage_n();
        let stage_shape_k = stage_config.elements_in_stage_k();

        let check_k_bounds = !(problem.k as u32).is_multiple_of(stage_shape_k);
        let check_m_bounds = !(problem.m as u32).is_multiple_of(stage_shape_m);
        let check_n_bounds = !(problem.n as u32).is_multiple_of(stage_shape_n);

        let num_planes = if !selection.load_specialization_config.has_specialization() {
            stage_config.num_main_flow_planes()
        } else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Specialization is unavailable for simple tma matmul.",
            )));
        };

        let plane_role_config = stage_config.plane_role_config();
        let num_stages = 1;
        let precompute_job = selection.loading_precompute_strategy.into();
        let reader_mode = selection.reader_mode;
        let plane_dim = selection.plane_dim;
        let specialization_tensor_config = SpecializationTensorConfig::MainFlowOnly;

        // Not used in simple
        let event_loading_mode = EventLoadingMode::Relaxed;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_k_bounds,
            matrix_layout: problem.lhs_layout,
            view_direction: ViewDirection::Col,
        };

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

        let lhs_smem_config = StageMemoryConfig {
            num_reading_planes: num_planes,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_m(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_k(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_m(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_k(),
            line_size: line_sizes.lhs as u32,
            matrix_layout: problem.lhs_layout,
            num_stages,
            swizzle: selection.shared_swizzle.lhs,
        };

        let rhs_smem_config = StageMemoryConfig {
            num_reading_planes: num_planes,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_k(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_n(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_k(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_n(),
            line_size: line_sizes.rhs as u32,
            matrix_layout: problem.rhs_layout,
            num_stages,
            swizzle: selection.shared_swizzle.rhs,
        };

        let out_smem_config = StageMemoryConfig {
            num_reading_planes: num_planes,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_m(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_n(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_m(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_n(),
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            num_stages,
            swizzle: selection.shared_swizzle.out,
        };

        let lhs_reader_config = GlobalReaderConfig {
            gmem_config: lhs_gmem_config,
            smem_config: lhs_smem_config,
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Lhs,
            event_loading_mode,
            specialization_tensor_config,
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
            specialization_tensor_config,
        };

        let writer_config = GlobalWriterConfig {
            gmem_config: out_gmem_config,
            smem_config: out_smem_config,
            role_rule_config: plane_role_config.rule,
            plane_dim: selection.plane_dim,
            num_partitions_n: selection.tiling_scheme.stage_partitions_in_stage_n(),
        };

        let config = SharedGlobalConfig {
            stage_config,
            num_planes,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
        };

        validate::<LL, RL, SMM::Config, R>(config, client, dtypes)
    }
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: SharedGlobalConfig<S>,
    client: &ComputeClient<R::Server>,
    dtypes: &MatmulElems,
) -> Result<SharedGlobalConfig<S>, MatmulSetupError> {
    LL::check::<R>(client, &config.lhs_reader_config, dtypes)?;
    RL::check::<R>(client, &config.rhs_reader_config, dtypes)?;
    cube_dim_validation(config.cube_dim())?;
    Ok(config)
}

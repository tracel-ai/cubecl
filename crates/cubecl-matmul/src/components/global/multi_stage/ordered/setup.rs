use crate::components::global::memory::{GlobalMemoryConfig, ViewDirection};
use crate::components::global::multi_stage::EventLoadingMode;
use crate::components::global::{
    GlobalConfig as _, GlobalReaderConfig, GlobalWriterConfig, MatmulPlaneCounts,
    SharedGlobalConfig, cube_dim_validation,
};
use crate::components::stage::TilingLayout;
use crate::components::stage::{StageConfig, StageMemoryConfig};
use crate::components::stage::{TilingLayoutConfig, TilingLayoutEnum};
use crate::components::{
    MatmulElems,
    global::{
        GlobalWriterFamily,
        read::{FullLoadingStrategy, PartialLoadingStrategy, sync::Synchronous},
    },
};
use crate::components::{MatmulLineSizes, MatmulSelection, MatrixLayout, StageIdent};
use crate::components::{MatmulPrecision, MatmulProblem, stage};
use crate::components::{
    TilingScheme,
    global::{
        WriteTiling,
        multi_stage::ordered::{LL, OrderedDoubleBufferingMatmul},
        read::LoadingValidation,
    },
};
use crate::components::{error::MatmulSetupError, stage::StridedStageFamily};
use crate::components::{global::GlobalMatmulFamily, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

/// Ordered double buffering matmul family for any precision
pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: PartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, RL, GW> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    RL: PartialLoadingStrategy<Stage = StridedStageFamily, SyncStrategy = Synchronous>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<
            MP,
            <LL as FullLoadingStrategy>::TilingLayout,
            RL::TilingLayout,
            NoTilingLayout,
            WriteTiling,
        >,
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
        let max_global_readers = selection
            .load_specialization_config
            .has_specialization()
            .then(|| {
                MaxGlobalReaderPlanes::new::<LL, RL>(
                    &selection.tiling_scheme,
                    line_sizes,
                    selection.plane_dim,
                )
            });

        let tiling_layout = TilingLayoutConfig {
            lhs: <LL as FullLoadingStrategy>::TilingLayout::to_enum(),
            rhs: RL::TilingLayout::to_enum(),
            acc: TilingLayoutEnum::Other,
            out: WriteTiling::to_enum(),
        };
        let stage_config = SMM::setup::<R>(
            client,
            problem,
            selection,
            line_sizes,
            tiling_layout,
            (1, 2).into(),
            max_global_readers,
            true,
            dtypes,
        )?;

        let plane_role_config = stage_config.plane_role_config();
        let plane_counts = MatmulPlaneCounts::new(
            selection.load_specialization_config,
            plane_role_config.plane_roles,
        );

        let stage_shape_m = stage_config.elements_in_stage_m();
        let stage_shape_n = stage_config.elements_in_stage_n();
        let stage_shape_k = stage_config.elements_in_stage_k();

        let check_m_bounds = !(problem.m as u32).is_multiple_of(stage_shape_m);
        let check_n_bounds = !(problem.n as u32).is_multiple_of(stage_shape_n);
        let check_k_bounds = !(problem.k as u32).is_multiple_of(2 * stage_shape_k);

        let precompute_job = selection.loading_precompute_strategy.into();
        let plane_dim = selection.plane_dim;
        let reader_mode = selection.reader_mode;

        let lhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.lhs as u32,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_k_bounds,
            matrix_layout: problem.lhs_layout,
            view_direction: ViewDirection::Col,
            stage_swizzle: todo!(),
        };

        let rhs_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.rhs as u32,
            check_row_bounds: check_k_bounds,
            check_col_bounds: check_n_bounds,
            matrix_layout: problem.rhs_layout,
            view_direction: ViewDirection::Row,
            stage_swizzle: todo!(),
        };

        let out_gmem_config = GlobalMemoryConfig {
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: check_m_bounds,
            check_col_bounds: check_n_bounds,
            view_direction: ViewDirection::None,
            stage_swizzle: todo!(),
        };

        let lhs_smem_config = StageMemoryConfig {
            num_reading_planes: plane_counts.lhs,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_m(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_k(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_m(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_k(),
            line_size: line_sizes.lhs as u32,
            matrix_layout: problem.lhs_layout,
            num_stages: 1,
            swizzle: todo!(),
        };

        let rhs_smem_config = StageMemoryConfig {
            num_reading_planes: plane_counts.rhs,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_k(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_n(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_k(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_n(),
            line_size: line_sizes.rhs as u32,
            matrix_layout: problem.rhs_layout,
            num_stages: 2,
            swizzle: todo!(),
        };

        let out_smem_config = StageMemoryConfig {
            num_reading_planes: plane_counts.out,
            elements_in_tile_row: selection.tiling_scheme.elements_in_tile_m(),
            elements_in_tile_col: selection.tiling_scheme.elements_in_tile_n(),
            tiles_in_stage_row: selection.tiling_scheme.tiles_in_stage_m(),
            tiles_in_stage_col: selection.tiling_scheme.tiles_in_stage_n(),
            line_size: line_sizes.out as u32,
            matrix_layout: MatrixLayout::RowMajor,
            num_stages: 1,
            swizzle: todo!(),
        };

        let lhs_reader_config = GlobalReaderConfig {
            gmem_config: lhs_gmem_config,
            smem_config: lhs_smem_config,
            precompute_job,
            plane_dim,
            plane_role_config,
            reader_mode,
            stage_ident: StageIdent::Lhs,
            event_loading_mode: EventLoadingMode::Ordered,
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
            event_loading_mode: EventLoadingMode::Relaxed,
            specialization_tensor_config: selection.load_specialization_config.rhs,
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
            num_planes: plane_counts.total,
            lhs_reader_config,
            rhs_reader_config,
            writer_config,
        };

        validate::<LL, RL, SMM::Config, R>(config, client, selection.tiling_scheme, dtypes)
    }
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: SharedGlobalConfig<S>,
    client: &ComputeClient<R::Server>,
    tiling_scheme: TilingScheme,
    dtypes: &MatmulElems,
) -> Result<SharedGlobalConfig<S>, MatmulSetupError> {
    LL::check::<R>(client, &config.lhs_reader_config, dtypes)?;
    RL::check::<R>(client, &config.rhs_reader_config, dtypes)?;
    cube_dim_validation(config.cube_dim())?;

    if tiling_scheme.stage_partitions_in_stage_n() > 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Ordered does not support number of stage partitions > 1 in n",
        )));
    }

    if config
        .lhs_reader_config
        .specialization_tensor_config
        .has_specialization()
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Error: In Ordered lhs loading cannot be outside of main flow",
        )));
    }

    Ok(config)
}

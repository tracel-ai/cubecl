use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_runtime::MmaConfig;

use crate::components::{MatmulProblem, MultiRowStrategy, SwizzleConfig, tile};
use crate::components::{
    adjust_dtypes,
    global::{
        multi_stage::specialized::SpecializedMatmulFamily,
        read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
    },
};
use crate::components::{batch::CubeCountPlanSelection, stage::PlaneMatmulFamily};
use crate::components::{
    batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    tile::io::{Filled, Strided},
};
use crate::components::{global::PlaneWriterFamily, stage::StageFamily};
use crate::components::{stage::FilledStageFamily, tile::TileMatmulFamily};
use crate::kernels::layered::algorithm::base;
use crate::kernels::layered::selector::{PlaneMatmulSelectionOptions, plane_matmul_selection};
use crate::{
    components::{
        MatmulElems, MatmulLineSizes, MatmulSelection, MatmulSetupError, MatrixLayout,
        TilingScheme,
        batch::{GlobalOrderSelection, HypercubeSelection, SmAllocation},
        global::{LoadSpecializationConfig, SpecializationTensorConfig},
        stage::PartitionBuffering,
    },
    kernels::layered::selector::select_swizzle,
};

/// Plane accelerated specialized matmul with TMA readers
pub struct SpecializedAlgorithm<TMM, L = AsyncPartialTmaLoading> {
    pub _phantom: PhantomData<(TMM, L)>,
}

impl<TMM, L> base::Algorithm for SpecializedAlgorithm<TMM, L>
where
    TMM: tile::TileMatmulFamily<
            LhsTile = <L::Stage as StageFamily>::TileKind,
            RhsTile = <L::Stage as StageFamily>::TileKind,
            AccTile = Filled,
            OutTile = Strided,
        >,
    L: AsyncPartialLoadingStrategy,
{
    type SelectionArgs = ();
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, L::Stage, L::Stage, FilledStageFamily>;
    type GlobalMatmul = SpecializedMatmulFamily<Self::StageMatmul, L, PlaneWriterFamily>;
    type BatchMatmul =
        PartitionedBatchMatmulFamily<Self::GlobalMatmul, RowMajorGlobalPartitionMatmul>;

    fn selection<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        _args: &Self::SelectionArgs,
        dtypes: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneMatmulSelectionOptions {
                specialized: true,
                multi_row_strategy: MultiRowStrategy::Adaptive {
                    minimum_stage_count: 8,
                },
                swizzled: TMM::should_swizzle(client),
                ..Default::default()
            },
        )
    }
}

#[allow(unused, reason = "needs more tuning")]
fn selection_specialized<R: Runtime, TMM: TileMatmulFamily>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    swizzle: bool,
    dtypes: &mut MatmulElems,
    line_sizes: &MatmulLineSizes,
) -> Result<MatmulSelection, MatmulSetupError> {
    adjust_dtypes(client, dtypes, TMM::requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        TMM::is_supported(
            client,
            MmaConfig {
                a_type: *dtypes.lhs_register,
                b_type: *dtypes.rhs_register,
                cd_type: *dtypes.acc_register,
                m,
                n,
                k,
            },
        )
    };
    let cube_count_plan = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountPlanSelection::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountPlanSelection::Flattened,
    };

    let tiling_scheme = if supported(16, 8, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 8, 16).into())
            .with_partition_size((1, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else if supported(16, 16, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 16, 16).into())
            .with_partition_size((1, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else {
        return plane_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            dtypes,
            line_sizes,
            PlaneMatmulSelectionOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        );
    };

    let hypercube = HypercubeSelection::builder(&tiling_scheme)
        .global_order(GlobalOrderSelection::SwizzleRow {
            m: problem.m as u32,
            w: 4,
        })
        .cube_count_plan(cube_count_plan)
        .build();

    let mut builder = MatmulSelection::builder(tiling_scheme, plane_dim)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_config(hypercube)
        .load_specialization_config(LoadSpecializationConfig {
            lhs: SpecializationTensorConfig::LoadFlowOnly,
            rhs: SpecializationTensorConfig::LoadFlowOnly,
        });

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m(),
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n(),
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k(),
        };

        let lhs = select_swizzle(lhs_swizzle_dim, *dtypes.lhs_stage, line_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, *dtypes.rhs_stage, line_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleConfig {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok(builder.build())
}

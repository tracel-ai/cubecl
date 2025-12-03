use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::batch::HypercubeBlueprint;
use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::components::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTileSize, AttentionTilingScheme,
};
use crate::kernels::SharedAttentionSettings;
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    kernels::Algorithm,
};

pub struct BlackboxAcceleratedAlgorithm {}

impl Algorithm for BlackboxAcceleratedAlgorithm {
    type TileAttention = BlackboxAcceleratedTileAttention;
    type StageAttention = PlanePartitionStageAttentionFamily<
        Self::TileAttention,
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    type Settings = SharedAttentionSettings;

    fn blueprint<R: cubecl_core::Runtime>(
        client: &ComputeClient<R>,
        problem: &AttentionProblem,
        settings: &Self::Settings,
    ) -> Result<AttentionBlueprint, AttentionSetupError> {
        #[cfg(target_os = "macos")]
        let tile_size = AttentionTileSize {
            seq_q: 8,
            head_dim: 8,
            seq_kv: 8,
            val_dim: 8,
        };
        #[cfg(not(target_os = "macos"))]
        let tile_size = AttentionTileSize {
            seq_q: 16,
            head_dim: 16,
            seq_kv: 16,
            val_dim: 16,
        };

        if problem.head_dim as u32 % tile_size.head_dim != 0 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Tile size head dim must divide problem head dim".to_string(),
            )));
        }

        let partition_head_dim = problem.head_dim as u32 / tile_size.head_dim;
        let partition_val_dim = partition_head_dim;

        let plane_dim = client.properties().hardware.plane_size_max;

        if partition_head_dim * tile_size.head_dim != problem.head_dim as u32 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Tiling scheme's total head dim must equal problem's head dim".to_string(),
            )));
        }

        let tiling_scheme = settings.tiling_scheme.unwrap_or(AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: partition_head_dim,
                seq_kv: 1,
                val_dim: partition_val_dim,
            },
            stage_size: AttentionStageSize { seq_q: 1 },
        });

        let num_planes = tiling_scheme.stage_size.seq_q
            * Self::TileAttention::computation_resources()?.num_planes(plane_dim)?;

        Ok(AttentionBlueprint {
            hypercube_blueprint: HypercubeBlueprint {},
            plane_dim,
            num_planes,
            reuse_key_value: settings.reuse_key_value,
            two_rows_in_array_tile: settings.two_rows_in_array_tile,
            line_sizes: problem.line_sizes.clone(),
            masked: problem.masked,
            causal: problem.causal,
            tiling_scheme,
            check_bounds: tiling_scheme.check_bounds(problem),
        })
    }

    fn dtypes<R: cubecl_core::Runtime>(
        _client: &ComputeClient<R>,
        problem: &AttentionProblem,
        _blueprint: &AttentionBlueprint,
    ) -> Result<AttentionElems, AttentionSetupError> {
        Ok(AttentionElems::from_problem(problem))
    }
}

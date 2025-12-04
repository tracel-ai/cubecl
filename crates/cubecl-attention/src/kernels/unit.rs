use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::ComputeResources;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::batch::HypercubeBlueprint;
use crate::components::stage::unit::UnitPartitionStageAttentionFamily;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::unit_register::UnitRegisterTileAttention;
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

pub struct UnitAlgorithm {}

impl Algorithm for UnitAlgorithm {
    type TileAttention = UnitRegisterTileAttention;
    type StageAttention = UnitPartitionStageAttentionFamily<
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
        let tile_size = AttentionTileSize {
            seq_q: 4,
            head_dim: 4,
            seq_kv: 4,
            val_dim: 4,
        };

        if problem.head_dim as u32 % tile_size.head_dim != 0 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Tile size head dim must divide problem head dim".to_string(),
            )));
        }

        let partition_head_dim = problem.head_dim as u32 / tile_size.head_dim;
        let partition_val_dim = partition_head_dim;

        let plane_dim = client.properties().hardware.plane_size_max;

        let tiling_scheme = settings.tiling_scheme.unwrap_or(AttentionTilingScheme {
            tile_size,
            partition_size: AttentionPartitionSize {
                seq_q: 1,
                head_dim: partition_head_dim,
                seq_kv: 1,
                val_dim: partition_val_dim,
            },
            stage_size: AttentionStageSize { seq_q: plane_dim },
        });

        let compute_resources = match Self::TileAttention::computation_resources()? {
            ComputeResources::Units(units) => {
                ComputeResources::Units(units * tiling_scheme.stage_size.seq_q)
            }
            _ => {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Error: Expected unit tile attention, got a plane tile attention".to_string(),
                )));
            }
        };

        let num_planes = compute_resources.num_planes(plane_dim)?;

        // Not sure where to put this, it depends on blueprint and problem
        if partition_head_dim * tile_size.head_dim != problem.head_dim as u32 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
                "Tiling scheme's total head dim must equal problem's head dim".to_string(),
            )));
        }

        Ok(AttentionBlueprint {
            hypercube_blueprint: HypercubeBlueprint {},
            tiling_scheme,
            plane_dim,
            num_planes,
            reuse_key_value: false,
            two_rows_in_array_tile: false,
            line_sizes: problem.line_sizes.clone(),
            masked: problem.masked,
            causal: problem.causal,
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

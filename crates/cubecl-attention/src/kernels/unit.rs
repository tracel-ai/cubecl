use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::batch::HypercubeSelection;
use crate::components::stage::unit::UnitPartitionStageAttentionFamily;
use crate::components::tile::unit_register::UnitRegisterTileAttention;
use crate::components::{
    AttentionElems, AttentionLineSizes, AttentionPartitionSize, AttentionProblem,
    AttentionSelection, AttentionSetupError, AttentionStageSize, AttentionTileSize,
    AttentionTilingScheme,
};
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

    fn selection<R: cubecl_core::Runtime>(
        _client: &ComputeClient<R>,
        problem: &AttentionProblem,
        plane_dim: u32,
        _line_sizes: &AttentionLineSizes,
        _dtypes: &AttentionElems,
    ) -> Result<AttentionSelection, AttentionSetupError> {
        let tile_size = AttentionTileSize {
            seq_q: 4,
            head_dim: 4,
            seq_kv: 4,
            val_dim: 4,
        };

        assert!(problem.head_dim as u32 % tile_size.head_dim == 0);
        let partition_head_dim = problem.head_dim as u32 / tile_size.head_dim;
        let partition_val_dim = partition_head_dim;

        Ok(AttentionSelection {
            hypercube_selection: HypercubeSelection {},
            tiling_scheme: AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: 1,
                    head_dim: partition_head_dim,
                    seq_kv: 1,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize { seq_q: plane_dim },
            },
            plane_dim,
            reuse_key_value: false,
            two_rows_in_array_tile: false,
        })
    }
}

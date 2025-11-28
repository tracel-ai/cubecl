use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::batch::HypercubeSelection;
use crate::components::stage::plane::PlanePartitionStageAttentionFamily;
use crate::components::tile::TileAttentionFamily;
use crate::components::tile::accelerated::BlackboxAcceleratedTileAttention;
use crate::components::{
    AttentionElems, AttentionLineSizes, AttentionPartitionSize, AttentionProblem,
    AttentionSelection, AttentionSetupError, AttentionStageSize, AttentionTileSize,
    AttentionTilingScheme,
};
use crate::{
    components::{
        AvailableLineSizes, batch::simple::SimpleBatchAttentionFamily,
        global::simple::SimpleGlobalAttentionFamily,
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

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        Self::TileAttention::filter_line_sizes(available_line_sizes)
    }

    fn selection<R: cubecl_core::Runtime>(
        _client: &ComputeClient<R>,
        problem: &AttentionProblem,
        plane_dim: u32,
        _line_sizes: &AttentionLineSizes,
        _dtypes: &AttentionElems,
    ) -> Result<AttentionSelection, AttentionSetupError> {
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
                stage_size: AttentionStageSize { seq_q: 1 },
            },
            plane_dim,
            reuse_key_value: false,
            two_rows_in_array_tile: false,
        })
    }
}

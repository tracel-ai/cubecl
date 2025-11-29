use cubecl_matmul::components::TileSize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionTilingScheme {
    pub tile_size: AttentionTileSize,
    pub partition_size: AttentionPartitionSize,
    pub stage_size: AttentionStageSize,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Score matmul: (seq_q, head_dim) @ (head_dim, seq_kv) → (seq_q, seq_kv)
// Value matmul: (seq_q, seq_kv) @ (seq_kv, val_dim) → (seq_q, val_dim)
pub struct AttentionTileSize {
    pub seq_q: u32,    // Query sequence length (m of both matmuls)
    pub head_dim: u32, // Head/embedding dimension, Shared Q-K dimension (k of score matmul)
    pub seq_kv: u32,   // Key/Value sequence length (n of score matmul, k of value matmul)
    pub val_dim: u32,  // Value output dimension (n of value matmul)
}

impl AttentionTileSize {
    pub fn to_score_matmul_tile_size(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.seq_kv,
            k: self.head_dim,
        }
    }

    pub fn to_value_matmul_tile_size(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.val_dim,
            k: self.seq_kv,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionPartitionSize {
    pub seq_q: u32,
    pub head_dim: u32,
    pub seq_kv: u32,
    pub val_dim: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionStageSize {
    // Other dims don't make sense
    pub seq_q: u32,
}

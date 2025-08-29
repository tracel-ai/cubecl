use cubecl_matmul::components::TileSize;

use crate::components::FlashIdent;

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
    pub fn query_size(&self) -> u32 {
        self.seq_q * self.head_dim
    }

    pub fn key_size(&self) -> u32 {
        self.head_dim * self.seq_kv
    }

    pub fn score_prob_size(&self) -> u32 {
        self.seq_q * self.seq_kv
    }

    pub fn value_size(&self) -> u32 {
        self.seq_kv * self.val_dim
    }

    pub fn accumulator_size(&self) -> u32 {
        self.seq_q * self.val_dim
    }

    pub fn num_rows(&self, ident: FlashIdent) -> u32 {
        match ident {
            FlashIdent::Query => self.seq_q,
            FlashIdent::Key => self.seq_kv,
            FlashIdent::ScoreProb => self.seq_q,
            FlashIdent::Value => self.seq_kv,
            FlashIdent::Mask => todo!(),
            FlashIdent::Out => self.seq_q,
        }
    }

    pub(crate) fn num_cols(&self, ident: FlashIdent) -> u32 {
        match ident {
            FlashIdent::Query => self.head_dim,
            FlashIdent::Key => self.head_dim,
            FlashIdent::ScoreProb => self.seq_kv,
            FlashIdent::Value => self.val_dim,
            FlashIdent::Mask => todo!(),
            FlashIdent::Out => self.val_dim,
        }
    }

    pub fn to_score_matmul(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.seq_kv,
            k: self.head_dim,
        }
    }

    pub fn to_value_matmul(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.val_dim,
            k: self.seq_kv,
        }
    }

    pub fn can_reuse_key_value(&self) -> bool {
        self.head_dim == self.val_dim
    }
}

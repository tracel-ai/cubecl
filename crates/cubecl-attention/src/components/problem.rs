use crate::components::AttentionIdent;

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionProblem {
    /// Batch size
    pub batch: usize,
    /// Number of attention heads
    pub num_heads: usize,

    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_kv: usize,
    /// Dimension of each head (d)
    pub head_dim: usize,
    /// Dimension of each value vector.  
    /// Usually equal to `head_dim`, but may differ in some variants
    pub val_dim: usize,

    /// Whether a mask is supplied (shape is always [batch, seq_q, heads, seq_kv])
    pub masked: bool,
    /// Whether there is a causal mask
    pub causal: bool,
}

impl AttentionProblem {
    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        match ident {
            AttentionIdent::Query => [self.batch, self.num_heads, self.seq_q, self.head_dim],
            AttentionIdent::Key => [self.batch, self.num_heads, self.seq_kv, self.head_dim],
            AttentionIdent::Value => [self.batch, self.num_heads, self.seq_kv, self.val_dim],
            AttentionIdent::Mask => [self.batch, self.num_heads, self.seq_q, self.seq_kv],
            AttentionIdent::Out => [self.batch, self.num_heads, self.seq_q, self.val_dim],
            AttentionIdent::Softmax => unreachable!("Not a materialized tensor"),
        }
    }
}

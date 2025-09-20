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

    /// Whether a mask is applied (shape is always [batch, seq_q, heads, seq_k])
    pub masked: bool,
}

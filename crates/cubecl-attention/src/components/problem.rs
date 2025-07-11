use cubecl_matmul::components::MatrixLayout;

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionProblem {
    /// Batch size
    pub batch: usize,
    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_k: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each head (d)
    pub head_dim: usize,

    /// Whether a mask is applied (shape is always [batch, seq_q, heads, seq_k])
    pub masked: bool,

    /// Memory layout of query
    pub q_layout: Option<MatrixLayout>,
    /// Memory layout of key
    pub k_layout: Option<MatrixLayout>,
    /// Memory layout of value
    pub v_layout: Option<MatrixLayout>,
}

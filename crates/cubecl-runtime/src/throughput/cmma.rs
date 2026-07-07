use cubecl_ir::ElemType;

/// Configuration for a matrix multiplication (CMMA) operation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeCmmaConfig {
    /// The data type used to store the running sum.
    pub accumulator_type: AccumulatorType,
    /// The spatial dimensions of the operation.
    pub matrix_sizes: MatrixSizes,
}

/// The element type of the accumulator.
pub type AccumulatorType = ElemType;

/// The M, N, and K dimensions of a matrix multiplication.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct MatrixSizes {
    /// Rows in the output matrix.
    pub m: usize,
    /// Columns in the output matrix.
    pub n: usize,
    /// The shared inner dimension of the input matrices.
    pub k: usize,
}

impl MatrixSizes {
    /// Returns the total iteration volume (M * N * K).
    pub fn num_elems(&self) -> usize {
        self.m * self.n * self.k
    }
}

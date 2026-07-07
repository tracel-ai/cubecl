use cubecl_ir::ElemType;

#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeCmmaConfig {
    pub accumulator_type: AccumulatorType,
    pub matrix_sizes: MatrixSizes,
}

pub type AccumulatorType = ElemType;

/// Defines the spatial dimensions for a matrix multiplication operation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct MatrixSizes {
    /// The number of rows in the output matrix and the first input matrix.
    pub m: usize,
    /// The number of columns in the output matrix and the second input matrix.
    pub n: usize,
    /// The inner dimension shared between the two input matrices.
    pub k: usize,
}

impl MatrixSizes {
    pub fn num_elems(&self) -> usize {
        self.m * self.n * self.k
    }
}

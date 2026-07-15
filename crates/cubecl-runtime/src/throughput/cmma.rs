use crate::{client::ComputeClient, runtime::Runtime};
use cubecl_ir::{ElemType, StorageType};

/// Configuration for a matrix multiplication (CMMA) operation.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeCmmaConfig {
    /// The data type used to store the running sum.
    pub accumulator_type: AccumulatorType,
    /// The spatial dimensions of the operation.
    pub cmma_dims: CmmaDims,
}

/// The element type of the accumulator.
pub type AccumulatorType = ElemType;

/// The M, N, and K dimensions of a matrix multiplication.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct CmmaDims {
    /// Rows in the output matrix.
    pub m: usize,
    /// Columns in the output matrix.
    pub n: usize,
    /// The shared inner dimension of the input matrices.
    pub k: usize,
}

impl CmmaDims {
    /// Returns the total iteration volume (M * N * K).
    pub fn num_elems(&self) -> usize {
        self.m * self.n * self.k
    }
}

/// Resolves the largest supported CMMA or MMA tile size `(m, n, k)`.
pub fn select_cmma_tile<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: StorageType,
    rhs: StorageType,
    acc: StorageType,
    (m, n, k): (usize, usize, usize),
) -> Option<(u32, u32, u32)> {
    let props = client.properties();

    props
        .features
        .matmul
        .cmma
        .iter()
        // Combine both CMMA and MMA supported hardware features.
        .chain(props.features.matmul.mma.iter())
        // Filter for instructions matching the exact input/accumulator data types.
        .filter(|it| it.a_type == lhs && it.b_type == rhs && it.cd_type == acc)
        // Ensure the hardware tile size actually fits within our problem dimensions.
        .filter(|it| m >= it.m as usize && n >= it.n as usize && k >= it.k as usize)
        // Select the tile with the largest volume to maximize throughput.
        .max_by_key(|it| it.m as u64 * it.n as u64 * it.k as u64)
        .map(|it| (it.m as u32, it.n as u32, it.k as u32))
}

use crate::client::Client;
use cubecl_ir::ElemType;

/// Resolves the largest supported CMMA or MMA tile size `(m, n, k)`.
pub fn select_cmma_tile(
    client: &Client,
    lhs: ElemType,
    rhs: ElemType,
    acc: ElemType,
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
        .map(|it| (it.m, it.n, it.k))
}

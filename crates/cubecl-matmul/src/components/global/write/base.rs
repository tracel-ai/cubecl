use crate::components::global::memory::GlobalMemoryConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::{Coordinates, Coords2d};

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<EO: Numeric>: CubeType + 'static + Send + Sync {
    /// Coordinates used to index the tensor
    type Coordinates: Coordinates;

    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write(
        this: &mut Self,
        out_smem_slice: Slice<Line<EO>>,
        tile: Coords2d,
        #[comptime] plane_dim: u32,
        #[comptime] config: GlobalMemoryConfig,
    );
}

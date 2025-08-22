use crate::components::global::GlobalConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coordinates;

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<EO: Numeric>: CubeType + 'static + Send + Sync {
    /// Coordinates used to index the tensor
    type Coordinates: Coordinates;

    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write<G: GlobalConfig>(
        this: &mut Self,
        slice: Slice<Line<EO>>,
        tile_m: u32,
        tile_n: u32,
        #[comptime] config: G,
    );
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_modular::matmul_global::GmmConfig;

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait StageWriter<EG: Numeric, G: GmmConfig>: CubeType + 'static + Send + Sync {
    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write<ES: Numeric>(
        this: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    );
}

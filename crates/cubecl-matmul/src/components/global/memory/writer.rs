use crate::components::global::memory::GlobalMemoryConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords3d};

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct TensorWriter<EO: Numeric> {
    pub view: View<Line<EO>, Coords3d, ReadWrite>,
}

unsafe impl<EG: Numeric> Sync for TensorWriter<EG> {}
unsafe impl<EG: Numeric> Send for TensorWriter<EG> {}

#[cube]
impl<EG: Numeric> TensorWriter<EG> {
    /// Instantiate a write view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(view: View<Line<EG>, Coords3d, ReadWrite>) -> Self {
        TensorWriter::<EG> { view }
    }

    /// Writes data into the tensor view at the specified coordinates (tile_x, tile_y).
    ///
    /// Each unit writes one line in a coalesced manner for improved efficiency, assuming row-major layout.
    pub fn write_coalesced(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        value: Line<EG>,
        #[comptime] out_config: GlobalMemoryConfig,
    ) {
        let tile_size_m = out_config.elements_in_tile_row;
        let tile_size_n = out_config.elements_in_tile_col;

        let view_x = tile_x * tile_size_m + unit_id / tile_size_n;
        let view_y = tile_y * tile_size_n + unit_id % tile_size_n;

        self.view
            .write_checked((0, view_x, view_y), Line::cast_from(value));
    }
}

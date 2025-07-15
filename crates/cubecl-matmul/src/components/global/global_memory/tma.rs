use crate::components::Ident;
use crate::components::InputIdent;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Uses a [`TensorMap`] to actually execute the load.
pub struct MappedTensorReader<EG: Numeric> {
    pub tensor: TensorMap<EG>,
    pub tile_x: u32,
    pub tile_y: u32,
    pub batch: u32,
}

unsafe impl<EG: Numeric> Sync for MappedTensorReader<EG> {}
unsafe impl<EG: Numeric> Send for MappedTensorReader<EG> {}

#[cube]
impl<EG: Numeric> MappedTensorReader<EG> {
    /// Instantiate a read view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(tensor: TensorMap<EG>, tile_x: u32, tile_y: u32, batch: u32) -> Self {
        MappedTensorReader::<EG> {
            tensor,
            tile_x,
            tile_y,
            batch,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32, #[comptime] ident: Ident) {
        match ident.as_input_ident() {
            InputIdent::Lhs => {
                self.tile_y += k_offset;
            }
            InputIdent::Rhs => {
                self.tile_x += k_offset;
            }
        }
    }
}

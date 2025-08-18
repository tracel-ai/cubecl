use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use cubecl_matmul::components::global::GlobalConfig;

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct BiasReader<E: Numeric> {
    pub tensor: VirtualTensor<E>,
    pub n_offset: u32,
    pub shape_n: u32,
}

unsafe impl<E: Numeric> Sync for BiasReader<E> {}
unsafe impl<E: Numeric> Send for BiasReader<E> {}

#[cube]
impl<E: Numeric> BiasReader<E> {
    /// Load the 1D bias into shared memory
    pub fn new(tensor: VirtualTensor<E>, n_offset: u32, shape_n: u32) -> BiasReader<E> {
        BiasReader::<E> {
            tensor,
            n_offset,
            shape_n,
        }
    }

    /// Load the 1D bias into shared memory
    pub fn load_simple<G: GlobalConfig>(
        &self,
        unit_id: u32,
        #[comptime] line_size: u32,
    ) -> Line<E> {
        let view_n = self.n_offset + unit_id;
        let read_pos = view_n / line_size;

        select(
            view_n < self.shape_n,
            self.read(read_pos),
            Line::empty(line_size).fill(E::from_int(0)),
        )
    }

    fn read(&self, position: u32) -> Line<E> {
        self.tensor.read(position)
    }
}

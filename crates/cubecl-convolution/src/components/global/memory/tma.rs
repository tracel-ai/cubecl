use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[derive(CubeType)]
/// A view of a feature map tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct Im2colTmaReader<E: Numeric> {
    pub tensor: TensorMap<E>,
    pub n_offset: u32,
    pub spatial_offsets: Sequence<u32>,
    pub k_offset: u32,
}

#[cube]
impl<E: Numeric> Im2colTmaReader<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tensor: VirtualTensor<E>,
        n_offset: u32,
        spatial_offsets: Sequence<u32>,
        k_offset: u32,
    ) -> Im2colTmaReader<E> {
        let map = tensor.as_tensor_map();

        Im2colTmaReader::<E> {
            tensor: map,
            n_offset,
            spatial_offsets,
            k_offset,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32) {
        self.k_offset += k_offset;
    }
}

unsafe impl<E: Numeric> Sync for Im2colTmaReader<E> {}
unsafe impl<E: Numeric> Send for Im2colTmaReader<E> {}

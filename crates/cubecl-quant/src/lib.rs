#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod qparams;

pub mod scheme;

#[cfg(feature = "export_tests")]
pub mod tests;

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use cubecl_core::{Runtime, client::ComputeClient, prelude::TensorHandleRef};
    use cubecl_std::tensor::StridedLayoutArgs;

    use crate::scheme::{QuantLevel, QuantScheme};

    pub(crate) fn strided_layout<'a, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tensor: &TensorHandleRef<R>,
    ) -> StridedLayoutArgs<'a, R> {
        let rank = tensor.shape.len();
        if rank <= 1 || tensor.shape[rank - 1] == tensor.strides[rank - 2] {
            StridedLayoutArgs::none()
        } else {
            StridedLayoutArgs::strided(client, tensor.shape[rank - 1] as u32)
        }
    }

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            assert!(
                *block_size % div == 0,
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }
}

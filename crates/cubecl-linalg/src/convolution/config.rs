use crate::matmul::components::global::GlobalConfig;

/// Convolution specific config, extends regular matmul [`Config`](global::Config)
pub trait ConvGemmConfig: GlobalConfig {
    /// The size of the convolution kernel at `dim`
    fn kernel_size(&self, dim: u32) -> u32;
    /// The dilation of the kernel at `dim`
    fn dilation(&self, dim: u32) -> u32;
    /// The stride of the kernel at `dim`
    fn stride(&self, dim: u32) -> u32;
    /// The padding of the kernel at `dim`
    fn padding(&self, dim: u32) -> i32;
    /// The number of stages in the convolution kernel
    fn num_stages(&self) -> u32;
}

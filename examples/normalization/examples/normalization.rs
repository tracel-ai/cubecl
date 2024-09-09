fn main() {
    #[cfg(feature = "cuda")]
    normalization::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    normalization::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

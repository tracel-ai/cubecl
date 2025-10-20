fn main() {
    #[cfg(feature = "cuda")]
    transpose::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    transpose::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

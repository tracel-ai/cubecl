fn main() {
    #[cfg(feature = "cuda")]
    fusing::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    fusing::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

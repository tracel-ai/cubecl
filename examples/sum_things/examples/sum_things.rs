fn main() {
    #[cfg(feature = "cuda")]
    sum_things::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    sum_things::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

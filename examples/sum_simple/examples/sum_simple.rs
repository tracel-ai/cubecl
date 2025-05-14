fn main() {
    #[cfg(feature = "cuda")]
    sum_simple::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    sum_simple::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

fn main() {
    #[cfg(feature = "cuda")]
    loop_unrolling::basic::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    loop_unrolling::basic::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}

fn main() {
    #[cfg(feature = "cuda")]
    gelu::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    gelu::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    gelu::launch::<cubecl::cpu::CpuRuntime>(&Default::default());
}

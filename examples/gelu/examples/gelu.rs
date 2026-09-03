fn main() {
    #[cfg(feature = "cuda")]
    gelu::launch(&cubecl::Device::Cuda(Default::default()));
    #[cfg(feature = "wgpu")]
    gelu::launch(&cubecl::Device::Wgpu(Default::default()));
    #[cfg(feature = "cpu")]
    gelu::launch(&cubecl::Device::Cpu(Default::default()));
}

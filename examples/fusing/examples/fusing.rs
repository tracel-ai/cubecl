fn main() {
    #[cfg(feature = "cuda")]
    fusing::launch(&cubecl::Device::Cuda(Default::default()));
    #[cfg(feature = "wgpu")]
    fusing::launch(&cubecl::Device::Wgpu(Default::default()));
}

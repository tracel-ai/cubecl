fn main() {
    #[cfg(feature = "cuda")]
    normalization::launch(&cubecl::Device::Cuda(Default::default()));
    #[cfg(feature = "wgpu")]
    normalization::launch(&cubecl::Device::Wgpu(Default::default()));
}

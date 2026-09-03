fn main() {
    #[cfg(feature = "cuda")]
    sum_things::launch(&cubecl::Device::Cuda(Default::default()));
    #[cfg(feature = "wgpu")]
    sum_things::launch(&cubecl::Device::Wgpu(Default::default()));
}

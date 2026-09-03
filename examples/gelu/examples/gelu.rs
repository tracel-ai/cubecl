use cubecl::Device;

fn main() {
    #[cfg(feature = "cuda")]
    gelu::launch(&Device::Cuda(Default::default()));
    #[cfg(feature = "wgpu")]
    gelu::launch(&Device::Wgpu(Default::default()));
    #[cfg(feature = "cpu")]
    gelu::launch(&Device::Cpu(Default::default()));
}

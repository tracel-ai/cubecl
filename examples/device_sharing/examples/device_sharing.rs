fn main() {
    #[cfg(feature = "wgpu")] {
        let context_wgpu = device_sharing::create_wgpu_context().expect("No wgpu context found");
        let device_cubecl_wgpu = cubecl::wgpu::init_existing_device(
            context_wgpu.adapter,
            context_wgpu.device,
            context_wgpu.queue,
            Default::default(),
        );
        device_sharing::assert_device_cubecl_wgpu_is_existing(&device_cubecl_wgpu);
        sum_things::launch::<cubecl::wgpu::WgpuRuntime>(&device_cubecl_wgpu);
    }
}

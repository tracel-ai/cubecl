fn main() {
    #[cfg(feature = "wgpu")]
    {
        let setup_shared = device_sharing::create_wgpu_setup_from_raw();
        let device_cubecl = cubecl::wgpu::init_device(setup_shared.clone(), Default::default());
        device_sharing::assert_wgpu_device_existing(&device_cubecl);
        sum_things::launch::<cubecl::wgpu::WgpuRuntime>(&device_cubecl);
    }
}

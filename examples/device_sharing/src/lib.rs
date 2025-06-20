#[cfg(feature = "wgpu")]
mod device_sharing_wgpu {
    use cubecl::wgpu::{AutoGraphicsApi, GraphicsApi, WgpuDevice, WgpuSetup};
    use wgpu::Features;

    pub fn create_wgpu_setup_from_raw() -> WgpuSetup {
        cubecl::future::block_on(create_wgpu_setup_from_raw_async())
    }

    pub async fn create_wgpu_setup_from_raw_async() -> WgpuSetup {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Failed to create wgpu adapter from instance");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Raw"),
                required_features: adapter
                    .features()
                    .difference(Features::MAPPABLE_PRIMARY_BUFFERS),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create wgpu device from adapter");

        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: AutoGraphicsApi::backend(),
        }
    }

    pub fn assert_wgpu_device_existing(device: &WgpuDevice) {
        assert!(
            matches!(device, cubecl::wgpu::WgpuDevice::Existing(_)),
            "device should be WgpuDevice::Existing"
        );
    }
}

#[cfg(feature = "wgpu")]
pub use device_sharing_wgpu::*;

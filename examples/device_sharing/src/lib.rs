use std::sync::Arc;

pub struct WgpuContext {
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

pub fn create_wgpu_context() -> Option<WgpuContext> {
    cubecl::future::block_on(create_wgpu_context_async())
}

pub async fn create_wgpu_context_async() -> Option<WgpuContext> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&Default::default()).await?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .ok()?;

    Some(WgpuContext {
        adapter: adapter.into(),
        device: device.into(),
        queue: queue.into(),
    })
}

#[cfg(feature = "wgpu")]
pub fn assert_device_cubecl_wgpu_is_existing(device: &cubecl::wgpu::WgpuDevice) {
    assert!(
        matches!(device, cubecl::wgpu::WgpuDevice::Existing(_)),
        "device should be WgpuDevice::Existing"
    );
}

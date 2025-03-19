use std::{borrow::Cow, sync::Arc};

use cubecl_core::{ExecutionMode, Feature, WgpuCompilationOptions, prelude::CompiledKernel};
use cubecl_runtime::DeviceProperties;
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderStages,
};

use crate::{AutoCompiler, AutoRepresentation, WgpuServer};

#[cfg(feature = "spirv")]
use super::vulkan;
use super::wgsl;

impl WgpuServer {
    pub fn create_pipeline(
        &mut self,
        kernel: CompiledKernel<AutoCompiler>,
        mode: ExecutionMode,
    ) -> Arc<ComputePipeline> {
        let module = match &kernel.repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => {
                let spirv = repr.assemble();
                unsafe {
                    self.device
                        .create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                            label: Some(&kernel.entrypoint_name),
                            source: Cow::Borrowed(&spirv),
                        })
                }
            }
            _ => {
                let source = &kernel.source;

                let checks = wgpu::ShaderRuntimeChecks {
                    // Cube does not need wgpu bounds checks - OOB behaviour is instead
                    // checked by cube (if enabled).
                    // This is because the WebGPU specification only makes loose guarantees that Cube can't rely on.
                    bounds_checks: false,
                    // Loop bounds are only checked in checked mode.
                    force_loop_bounding: mode == ExecutionMode::Checked,
                };

                // SAFETY: Cube guarantees OOB safety when launching in checked mode. Launching in unchecked mode
                // is only available through the use of unsafe code.
                unsafe {
                    self.device.create_shader_module_trusted(
                        ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                        },
                        checks,
                    )
                }
            }
        };
        let bindings = match &kernel.repr {
            Some(AutoRepresentation::Wgsl(repr)) => Some(wgsl::bindings(repr)),
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => Some(vulkan::bindings(repr)),
            _ => None,
        };
        let layout = bindings.map(|bindings| {
            let bindings = bindings
                .into_iter()
                .map(|(i, _visibility)| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        #[cfg(not(exclusive_memory_only))]
                        ty: BufferBindingType::Storage { read_only: false },
                        #[cfg(exclusive_memory_only)]
                        ty: BufferBindingType::Storage {
                            read_only: matches!(
                                _visibility,
                                cubecl_core::compute::Visibility::Read
                            ),
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>();
            let layout = self
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bindings,
                });
            self.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                })
        });

        Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&kernel.entrypoint_name),
                    layout: layout.as_ref(),
                    module: &module,
                    entry_point: Some(&kernel.entrypoint_name),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                }),
        )
    }
}

#[cfg(not(feature = "spirv"))]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    wgsl::request_device(adapter).await
}

#[cfg(feature = "spirv")]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    if is_vulkan(adapter) {
        vulkan::request_vulkan_device(adapter).await
    } else {
        wgsl::request_device(adapter).await
    }
}

#[cfg(not(feature = "spirv"))]
pub fn register_features(
    _adapter: &Adapter,
    props: &mut DeviceProperties<Feature>,
    _comp_options: &mut WgpuCompilationOptions,
) {
    wgsl::register_types(props);
}

#[cfg(feature = "spirv")]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties<Feature>,
    comp_options: &mut WgpuCompilationOptions,
) {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options);
    } else {
        wgsl::register_types(props);
    }
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan, _, _>(|adapter| adapter.is_some()) }
}

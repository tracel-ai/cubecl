use super::wgsl;
use crate::{AutoCompiler, AutoRepresentation, WgpuServer};
use cubecl_core::{ExecutionMode, WgpuCompilationOptions, prelude::CompiledKernel};
use cubecl_ir::DeviceProperties;
use cubecl_runtime::compiler::CompilationError;
use std::{borrow::Cow, sync::Arc};
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderStages,
};

#[cfg(not(target_family = "wasm"))]
use crate::errors::{fetch_error, track_error};

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;
#[cfg(all(feature = "msl", target_os = "macos"))]
use cubecl_cpp::metal as cpp_metal;

impl WgpuServer {
    pub fn create_pipeline(
        &mut self,
        kernel: CompiledKernel<AutoCompiler>,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, CompilationError> {
        let module = match &kernel.repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => {
                let spirv = repr.assemble();
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough::SpirV(
                            wgpu::ShaderModuleDescriptorSpirV {
                                label: Some(&kernel.entrypoint_name),
                                source: Cow::Borrowed(&spirv),
                            },
                        ),
                    )
                }
            }
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentation::Msl(repr)) => {
                let source = &kernel.source;
                unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough::Msl(
                            wgpu::ShaderModuleDescriptorMsl {
                                entry_point: kernel.entrypoint_name.clone(),
                                label: Some(&kernel.entrypoint_name),
                                source: Cow::Borrowed(source),
                                num_workgroups: (repr.cube_dim.x, repr.cube_dim.y, repr.cube_dim.z),
                            },
                        ),
                    )
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

                #[cfg(not(target_family = "wasm"))]
                track_error(&self.device, wgpu::ErrorFilter::Validation);

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

        #[cfg(not(target_family = "wasm"))]
        if let Some(err) = cubecl_common::future::block_on(fetch_error(&self.device)) {
            return Err(CompilationError::Generic {
                reason: format!("{err}"),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }

        let bindings_info = match &kernel.repr {
            Some(AutoRepresentation::Wgsl(repr)) => Some(wgsl::bindings(repr)),
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentation::Msl(repr)) => Some(cpp_metal::bindings(repr)),
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => Some(vulkan::bindings(repr)),
            _ => None,
        };

        let layout = bindings_info.map(|bindings| {
            let (mut bindings, meta) = bindings;
            // When slices are shared, it needs to be read-write if ANY of the slices is read-write,
            // and since we can't be sure, we'll assume everything is read-write.
            if !cfg!(exclusive_memory_only) {
                bindings.fill(cubecl_runtime::kernel::Visibility::ReadWrite);
            }

            let bindings = bindings
                .into_iter()
                .chain(meta)
                .enumerate()
                .map(|(i, visibility)| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage {
                            read_only: matches!(
                                visibility,
                                cubecl_runtime::kernel::Visibility::Read
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

        let pipeline = self
            .device
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
            });
        Ok(Arc::new(pipeline))
    }
}

#[cfg(all(not(feature = "spirv"), not(feature = "msl")))]
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

#[cfg(all(feature = "msl", target_os = "macos"))]
pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    use super::metal;

    if is_metal(adapter) {
        metal::request_metal_device(adapter).await
    } else {
        panic!("metal device not found!");
    }
}

#[cfg(all(not(feature = "spirv"), not(feature = "msl")))]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options);
    } else {
        wgsl::register_wgsl_features(adapter, props, comp_options);
    }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options);
    } else {
        panic!("metal device not found!");
    }
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>().is_some() }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal>().is_some() }
}

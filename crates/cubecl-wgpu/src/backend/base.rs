use super::wgsl;
use crate::AutoRepresentationRef;
use crate::WgpuServer;
use cubecl_core::MemoryConfiguration;
use cubecl_core::{
    ExecutionMode, WgpuCompilationOptions, hash::StableHash, server::KernelArguments,
};
use cubecl_ir::DeviceProperties;
use cubecl_runtime::{compiler::CompilationError, id::KernelId};
use std::{borrow::Cow, sync::Arc};
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderModule, ShaderModuleDescriptor,
    ShaderStages,
};

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;
#[cfg(all(feature = "msl", target_os = "macos"))]
use cubecl_cpp::metal as cpp_metal;

impl WgpuServer {
    /// Loads a cached kernel if present and creates the pipeline for it.
    /// Returns `None` if the cache isn't enabled, `Some(Ok(pipeline))` if a cache entry was found,
    /// and `Some(Err(cache_key))` if the cache is enabled but doesn't contain this kernel.
    #[allow(
        clippy::type_complexity,
        reason = "required because of error propagation"
    )]
    #[allow(unused_variables)]
    pub fn load_cached_pipeline(
        &self,
        kernel_id: &KernelId,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Option<Result<Arc<ComputePipeline>, (u64, StableHash)>>, CompilationError> {
        #[cfg(not(feature = "spirv"))]
        let res = Ok(None);
        #[cfg(feature = "spirv")]
        let res = if let Some(cache) = &self.spirv_cache {
            let key = (self.utilities.properties_hash, kernel_id.stable_hash());
            if let Some(entry) = cache.get(&key) {
                log::trace!("Using SPIR-V cache");

                let repr = AutoRepresentationRef::SpirV(&entry.kernel);
                let module = self.create_module(&entry.entrypoint_name, Some(repr), "", mode)?;
                let pipeline =
                    self.create_pipeline(&entry.entrypoint_name, Some(repr), module, bindings);
                Ok(Some(Ok(pipeline)))
            } else {
                Ok(Some(Err(key)))
            }
        } else {
            Ok(None)
        };

        res
    }

    pub fn create_module(
        &self,
        entrypoint_name: &str,
        repr: Option<AutoRepresentationRef<'_>>,
        source: &str,
        mode: ExecutionMode,
    ) -> Result<ShaderModule, CompilationError> {
        #[allow(unused_assignments)]
        #[cfg(not(target_family = "wasm"))]
        let mut error_scope = None;

        match repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => unsafe {
                Ok(self.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some(entrypoint_name),
                        spirv: Some(Cow::Borrowed(&repr.assembled_module)),
                        ..Default::default()
                    },
                ))
            },
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentationRef::Msl(repr)) => unsafe {
                Ok(self.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some(entrypoint_name),
                        msl: Some(Cow::Borrowed(source)),
                        num_workgroups: (repr.cube_dim.x, repr.cube_dim.y, repr.cube_dim.z),
                        ..Default::default()
                    },
                ))
            },
            _ => {
                let _ = entrypoint_name; // otherwise unused
                let checks = wgpu::ShaderRuntimeChecks {
                    // Cube does not need wgpu bounds checks - OOB behaviour is instead
                    // checked by cube (if enabled).
                    // This is because the WebGPU specification only makes loose guarantees that Cube can't rely on.
                    bounds_checks: false,
                    // Loop bounds are only checked in checked mode.
                    force_loop_bounding: mode == ExecutionMode::Checked,
                    ..wgpu::ShaderRuntimeChecks::unchecked()
                };

                #[cfg(not(target_family = "wasm"))]
                {
                    error_scope = Some(self.device.push_error_scope(wgpu::ErrorFilter::Validation));
                }

                // SAFETY: Cube guarantees OOB safety when launching in checked mode. Launching in unchecked mode
                // is only available through the use of unsafe code.
                let module = unsafe {
                    self.device.create_shader_module_trusted(
                        ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                        },
                        checks,
                    )
                };

                #[cfg(not(target_family = "wasm"))]
                if let Some(scope) = error_scope
                    && let Some(err) = cubecl_common::future::block_on(scope.pop())
                {
                    return Err(CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: cubecl_common::backtrace::BackTrace::capture(),
                    });
                }

                Ok(module)
            }
        }
    }

    #[allow(unused_variables)]
    pub fn create_pipeline(
        &self,
        entrypoint_name: &str,
        repr: Option<AutoRepresentationRef<'_>>,
        module: ShaderModule,
        bindings: &KernelArguments,
    ) -> Arc<ComputePipeline> {
        let bindings_info = match repr {
            Some(AutoRepresentationRef::Wgsl(repr)) => Some(wgsl::bindings(repr, bindings)),
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentationRef::Msl(repr)) => Some(cpp_metal::bindings(repr, bindings)),
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => Some(vulkan::bindings(repr, bindings)),
            _ => None,
        };

        let layout = bindings_info.map(|bindings| {
            let (mut bindings, info, uniform_info) = bindings;
            // When slices are shared, it needs to be read-write if ANY of the slices is read-write,
            // and since we can't be sure, we'll assume everything is read-write.
            if !cfg!(exclusive_memory_only) {
                bindings.fill(cubecl_runtime::kernel::Visibility::ReadWrite);
            }

            let info = info.map(|_| match uniform_info {
                true => BufferBindingType::Uniform,
                false => BufferBindingType::Storage { read_only: true },
            });

            let bindings = bindings
                .into_iter()
                .map(|visibility| BufferBindingType::Storage {
                    read_only: matches!(visibility, cubecl_runtime::kernel::Visibility::Read),
                })
                .chain(info)
                .enumerate()
                .map(|(i, ty)| BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty,
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
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                })
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entrypoint_name),
                layout: layout.as_ref(),
                module: &module,
                entry_point: Some(entrypoint_name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });
        Arc::new(pipeline)
    }
}

pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    if let Some(result) = request_vulkan_device(adapter).await {
        return result;
    }
    if let Some(result) = request_metal_device(adapter).await {
        return result;
    }
    wgsl::request_device(adapter).await
}

#[cfg(feature = "spirv")]
async fn request_vulkan_device(adapter: &Adapter) -> Option<(Device, Queue)> {
    if is_vulkan(adapter) {
        vulkan::request_vulkan_device(adapter).await
    } else {
        None
    }
}

#[cfg(not(feature = "spirv"))]
async fn request_vulkan_device(_adapter: &Adapter) -> Option<(Device, Queue)> {
    None
}

#[cfg(all(feature = "msl", target_os = "macos"))]
async fn request_metal_device(adapter: &Adapter) -> Option<(Device, Queue)> {
    if is_metal(adapter) {
        Some(metal::request_metal_device(adapter).await)
    } else {
        None
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
async fn request_metal_device(_adapter: &Adapter) -> Option<(Device, Queue)> {
    None
}

pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) {
    if register_vulkan_features(adapter, props, comp_options, memory_config) {
        return;
    }
    if register_metal_features(adapter, props, comp_options, memory_config) {
        return;
    }
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_vulkan_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options, memory_config)
    } else {
        false
    }
}

#[cfg(not(feature = "spirv"))]
pub fn register_vulkan_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_metal_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options);
        true
    } else {
        false
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
pub fn register_metal_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>().is_some() }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal>().is_some() }
}

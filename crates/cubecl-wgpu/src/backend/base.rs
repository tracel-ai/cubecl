use super::wgsl;
use crate::WgpuServer;
use crate::{AutoRepresentationRef, CompilerInfo};
use cubecl_core::{
    CubeDim, ExecutionMode, WgpuCompilationOptions, hash::StableHash, server::KernelArguments,
};
use cubecl_core::{MemoryConfiguration, prelude::Visibility};
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
    ) -> Result<
        Option<Result<(Arc<ComputePipeline>, CompilerInfo), (u64, StableHash)>>,
        CompilationError,
    > {
        #[cfg(not(feature = "spirv"))]
        let res = Ok(None);
        #[cfg(feature = "spirv")]
        let res = if let Some(cache) = &self.spirv_cache {
            let key = (self.utilities.properties_hash, kernel_id.stable_hash());
            if let Some(entry) = cache.get(&key) {
                use crate::ParamsTransfer;

                log::trace!("Using SPIR-V cache");

                let params_transfer = match entry.kernel.immediate_size {
                    Some(_) => ParamsTransfer::Immediate,
                    None => ParamsTransfer::Uniform,
                };
                let repr = AutoRepresentationRef::SpirV(&entry.kernel);
                let module = self.create_module(
                    &entry.entrypoint_name,
                    kernel_id.cube_dim,
                    Some(repr),
                    "",
                    mode,
                )?;
                let pipeline =
                    self.create_pipeline(&entry.entrypoint_name, Some(repr), module, bindings);
                Ok(Some(Ok((
                    pipeline,
                    CompilerInfo::Vulkan { params_transfer },
                ))))
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
        cube_dim: CubeDim,
        repr: Option<AutoRepresentationRef<'_>>,
        source: &str,
        mode: ExecutionMode,
    ) -> Result<ShaderModule, CompilationError> {
        match repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => unsafe {
                Ok(self.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some(entrypoint_name),
                        spirv: Some(Cow::Borrowed(&repr.assembled_module)),
                        entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                            name: entrypoint_name.into(),
                            workgroup_size: cube_dim.into(),
                        }]),
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
                        entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                            name: entrypoint_name.into(),
                            workgroup_size: cube_dim.into(),
                        }]),
                        ..Default::default()
                    },
                ))
            },
            _ => {
                let _ = cube_dim;
                let checks = wgpu::ShaderRuntimeChecks {
                    // Cube does not need wgpu bounds checks - OOB behaviour is instead
                    // checked by cube (if enabled).
                    // This is because the WebGPU specification only makes loose guarantees that Cube can't rely on.
                    bounds_checks: false,
                    // Loop bounds are only checked in checked mode.
                    force_loop_bounding: mode == ExecutionMode::Checked,
                    ..wgpu::ShaderRuntimeChecks::unchecked()
                };

                log::trace!("[cubecl-wgpu] compiling WGSL module `{entrypoint_name}`\n{source}");

                let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);

                // SAFETY: Cube guarantees OOB safety when launching in checked mode. Launching in unchecked mode
                // is only available through the use of unsafe code.
                let module = unsafe {
                    self.device.create_shader_module_trusted(
                        ShaderModuleDescriptor {
                            label: Some(entrypoint_name),
                            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                        },
                        checks,
                    )
                };

                // `pop()` detaches from the LIFO stack immediately; only the
                // result is async. Safe to interleave with other push/pops.
                let err_future = error_scope.pop();

                #[cfg(not(target_family = "wasm"))]
                if let Some(err) = cubecl_common::future::block_on(err_future) {
                    log::error!(
                        "[cubecl-wgpu] WGSL compilation failed for kernel `{entrypoint_name}`:\n{err}\n--- shader source ({} bytes) ---\n{source}\n--- end shader ---",
                        source.len()
                    );
                    return Err(CompilationError::Generic {
                        reason: format!(
                            "WGSL compilation failed for kernel `{entrypoint_name}`: {err}"
                        ),
                        backtrace: cubecl_common::backtrace::BackTrace::capture(),
                    });
                }

                // On wasm we can't block; spawn a task that awaits the pop
                // future and logs.
                #[cfg(target_family = "wasm")]
                {
                    let entrypoint_name = entrypoint_name.to_string();
                    let source = source.to_string();
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Some(err) = err_future.await {
                            log::error!(
                                "[cubecl-wgpu] WGSL compilation failed for kernel `{entrypoint_name}`:\n{err}\n--- shader source ({} bytes) ---\n{source}\n--- end shader ---",
                                source.len()
                            );
                        }
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

        let layout = bindings_info.map(|(bindings, immediate_size)| {
            if !bindings.is_empty() {
                let bindings = bindings
                    .into_iter()
                    .map(|visibility| match visibility {
                        Visibility::Uniform => BufferBindingType::Uniform,
                        Visibility::Read => BufferBindingType::Storage { read_only: true },
                        Visibility::ReadWrite => BufferBindingType::Storage { read_only: false },
                    })
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
                        immediate_size: immediate_size as u32,
                    })
            } else {
                self.device
                    .create_pipeline_layout(&PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[],
                        immediate_size: immediate_size as u32,
                    })
            }
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

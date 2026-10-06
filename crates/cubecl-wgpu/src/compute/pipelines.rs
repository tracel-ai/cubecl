//! The half of wgpu's compilation that runs where the server does: the SPIR-V
//! store, and the pipelines the device builds from what it and the compiler
//! produce.

use alloc::sync::Arc;
use std::borrow::Cow;

use crate::backend::ModuleSource;
use crate::{AutoRepresentationRef, PipelineEntry, WgpuCompiler};
use cubecl_core::server::MetadataBindingInfo;
use cubecl_core::{CubeDim, ExecutionMode, prelude::Visibility};
#[cfg(feature = "spirv")]
use cubecl_environment::persistence::Store;
use cubecl_server::compiler::{ArtifactId, CompilationError, CompilationTarget};
#[cfg(feature = "spirv")]
use cubecl_server::compiler::{KernelCacheKey, store_compiled};
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderStages,
};

use super::artifact::{WgpuArtifact, WgpuArtifactCompiler};
use crate::backend::wgsl;

#[cfg(feature = "spirv")]
use crate::backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use crate::backend::metal;

/// What a launch's metadata binding is, as far as the pipeline layout that
/// receives it is concerned: the part of a launch, beyond its kernel, that a
/// pipeline is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataLayout {
    /// The launch passes no metadata, so the layout has no binding for it.
    Absent,
    /// Every value is known when the kernel is compiled; a backend that can
    /// may bind it as a uniform.
    Static,
    /// Part of it is sized at launch, so it is bound as a storage buffer.
    Dynamic,
}

impl MetadataLayout {
    /// The layout `info` is bound with.
    pub fn of(info: &MetadataBindingInfo) -> Self {
        if info.data.is_empty() {
            Self::Absent
        } else if info.dynamic_metadata_offset >= info.data.len() {
            Self::Static
        } else {
            Self::Dynamic
        }
    }
}

/// Builds wgpu pipelines for one device, from the SPIR-V store when it holds
/// the kernel and from [`WgpuArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct WgpuPipelines<C> {
    compiler: WgpuArtifactCompiler<C>,
    device: wgpu::Device,
    #[cfg(feature = "spirv")]
    spirv_store: Option<Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>>,
    /// What the SPIR-V store's keys are scoped by beside the kernel: SPIR-V
    /// compiled for one set of device properties is not valid for another.
    #[cfg(feature = "spirv")]
    properties_hash: u64,
    #[cfg(feature = "spirv")]
    build_id: cubecl_common::hash::StableHash,
}

impl<C> WgpuPipelines<C> {
    /// Pipelines built on `device`, with nothing built yet.
    pub fn new(
        compiler: WgpuArtifactCompiler<C>,
        device: wgpu::Device,
        #[cfg(feature = "spirv")] spirv_store: Option<
            Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>,
        >,
        #[cfg(feature = "spirv")] properties_hash: u64,
    ) -> Self {
        Self {
            compiler,
            device,
            #[cfg(feature = "spirv")]
            spirv_store,
            #[cfg(feature = "spirv")]
            properties_hash,
            #[cfg(feature = "spirv")]
            build_id: cubecl_server::compiler::build_id_hash(),
        }
    }

    fn create_module(
        &self,
        entrypoint_name: &str,
        cube_dim: CubeDim,
        source: ModuleSource<'_>,
        mode: ExecutionMode,
    ) -> Result<ShaderModule, CompilationError> {
        match source {
            #[cfg(feature = "spirv")]
            ModuleSource::SpirV(repr) => self
                .validated(|| unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some(entrypoint_name),
                            spirv: Some(Cow::Borrowed(&repr.assembled_module)),
                            entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                                name: entrypoint_name.into(),
                                workgroup_size: cube_dim.into(),
                            }]),
                            ..Default::default()
                        },
                    )
                })
                .map_err(|err| refused("SPIR-V module", entrypoint_name, err)),
            #[cfg(all(feature = "msl", target_os = "macos"))]
            ModuleSource::Msl(source) => self
                .validated(|| unsafe {
                    self.device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some(entrypoint_name),
                            msl: Some(Cow::Borrowed(source)),
                            entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                                name: entrypoint_name.into(),
                                workgroup_size: cube_dim.into(),
                            }]),
                            ..Default::default()
                        },
                    )
                })
                .map_err(|err| refused("MSL module", entrypoint_name, err)),
            ModuleSource::Wgsl(source) => {
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
                if let Some(err) = cubecl_environment::future::block_on(err_future) {
                    log::error!(
                        "[cubecl-wgpu] WGSL compilation failed for kernel `{entrypoint_name}`:\n{err}\n--- shader source ({} bytes) ---\n{source}\n--- end shader ---",
                        source.len()
                    );
                    return Err(CompilationError::Generic {
                        reason: format!(
                            "WGSL compilation failed for kernel `{entrypoint_name}`: {err}"
                        ),
                        backtrace: cubecl_environment::backtrace::BackTrace::capture(),
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
    fn create_pipeline(
        &self,
        entrypoint_name: &str,
        repr: Option<AutoRepresentationRef<'_>>,
        module: ShaderModule,
        metadata: MetadataLayout,
    ) -> Result<Arc<ComputePipeline>, CompilationError> {
        let bindings_info = match repr {
            Some(AutoRepresentationRef::Wgsl(repr)) => Some(wgsl::bindings(repr, metadata)),
            #[cfg(all(feature = "msl", target_os = "macos"))]
            Some(AutoRepresentationRef::Msl(repr)) => Some(metal::bindings(repr, metadata)),
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => Some(vulkan::bindings(repr)),
            _ => None,
        };

        let create = || {
            let layout = bindings_info.map(|(bindings, immediate_size)| {
                if !bindings.is_empty() {
                    let bindings = bindings
                        .into_iter()
                        .map(|visibility| match visibility {
                            Visibility::Uniform => BufferBindingType::Uniform,
                            Visibility::Read => BufferBindingType::Storage { read_only: true },
                            Visibility::ReadWrite => {
                                BufferBindingType::Storage { read_only: false }
                            }
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
        };
        self.validated(create)
            .map_err(|err| refused("pipeline", entrypoint_name, err))
    }

    /// Creates a device object under validation and internal error scopes, and returns what the
    /// device reported against it.
    ///
    /// wgpu reports a creation it refuses as an uncaptured device error, which panics the thread
    /// polling the device and leaves a read of the launch's outputs returning whatever they held.
    /// The scopes make the refusal the launch's error, which the outputs then carry. Blocks until
    /// the device has validated the creation; on wasm, which cannot block, the creation is
    /// unchecked.
    fn validated<T>(&self, create: impl FnOnce() -> T) -> Result<T, wgpu::Error> {
        #[cfg(target_family = "wasm")]
        return Ok(create());

        #[cfg(not(target_family = "wasm"))]
        {
            // A passthrough module the backend fails to compile is reported as internal, a
            // layout past the device's limits as validation. Both are refusals.
            let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
            let created = create();
            let internal = internal.pop();
            let validation = validation.pop();
            let refusal =
                cubecl_environment::future::block_on(async { internal.await.or(validation.await) });
            match refusal {
                Some(err) => Err(err),
                None => Ok(created),
            }
        }
    }
}

impl<C: WgpuCompiler> CompilationTarget for WgpuPipelines<C> {
    type Compiler = WgpuArtifactCompiler<C>;
    type Loaded = PipelineEntry;

    fn compiler(&self) -> &WgpuArtifactCompiler<C> {
        &self.compiler
    }

    /// WGSL is compiled by the driver on every run, so without the SPIR-V
    /// store there is nothing persisted for a switch to invalidate.
    fn persists(&self) -> bool {
        #[cfg(feature = "spirv")]
        return self.spirv_store.is_some();
        #[cfg(not(feature = "spirv"))]
        return false;
    }

    #[allow(unused_variables)]
    fn stored(&mut self, id: &ArtifactId<MetadataLayout>) -> Option<WgpuArtifact> {
        #[cfg(not(feature = "spirv"))]
        return None;

        #[cfg(feature = "spirv")]
        {
            use crate::{AutoRepresentation, CompilerInfo, ParamsTransfer};

            let key = (
                self.properties_hash,
                KernelCacheKey::new(&id.kernel, self.build_id),
            );
            let entry = self.spirv_store.as_mut()?.remove(&key)?;
            log::trace!("Using SPIR-V cache");

            let params_transfer = match entry.kernel.immediate_size {
                Some(_) => ParamsTransfer::Immediate,
                None => ParamsTransfer::Uniform,
            };
            Some(WgpuArtifact {
                entrypoint_name: entry.entrypoint_name,
                // SPIR-V reaches the device as an assembled module, never as text.
                source: String::new(),
                lang: "spv",
                io: entry.kernel.io.clone().map(Arc::from),
                repr: Some(AutoRepresentation::SpirV(entry.kernel)),
                compiler_info: CompilerInfo::Vulkan { params_transfer },
            })
        }
    }

    fn load(
        &mut self,
        id: &ArtifactId<MetadataLayout>,
        artifact: &WgpuArtifact,
    ) -> Result<PipelineEntry, CompilationError> {
        let repr = artifact.repr.as_ref().map(|repr| repr.as_ref());
        let module = self.create_module(
            &artifact.entrypoint_name,
            id.kernel.cube_dim.into(),
            ModuleSource::resolve(repr, artifact.lang, &artifact.source)?,
            id.kernel.mode,
        )?;
        let pipeline = self.create_pipeline(&artifact.entrypoint_name, repr, module, id.variant)?;
        Ok((pipeline, artifact.compiler_info, artifact.io.clone()))
    }

    /// Only a SPIR-V kernel is stored: any other build changes nothing.
    #[allow(unused_variables)]
    fn store(
        &mut self,
        id: &ArtifactId<MetadataLayout>,
        artifact: WgpuArtifact,
        _source: Option<&str>,
    ) -> bool {
        #[cfg(not(feature = "spirv"))]
        return false;

        #[cfg(feature = "spirv")]
        {
            let (Some(store), Some(crate::AutoRepresentation::SpirV(kernel))) =
                (self.spirv_store.as_mut(), artifact.repr)
            else {
                return false;
            };
            let key = (
                self.properties_hash,
                KernelCacheKey::new(&id.kernel, self.build_id),
            );
            store_compiled(
                store,
                key,
                cubecl_spirv::SpirvCacheEntry::new(artifact.entrypoint_name, kernel),
            )
        }
    }
}

/// The error a launch returns when the device refuses to create one of its objects.
fn refused(object: &str, entrypoint_name: &str, err: wgpu::Error) -> CompilationError {
    log::error!(
        "[cubecl-wgpu] the device refused the {object} of kernel `{entrypoint_name}`: {err}"
    );
    CompilationError::Generic {
        reason: format!("the device refused the {object} of kernel `{entrypoint_name}`: {err}"),
        backtrace: cubecl_environment::backtrace::BackTrace::capture(),
    }
}

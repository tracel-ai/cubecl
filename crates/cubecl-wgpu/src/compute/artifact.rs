//! The half of wgpu's compilation any thread can do: from a kernel to the
//! shader, and the pipeline the device builds from it.

use alloc::sync::Arc;
use core::marker::PhantomData;
use std::borrow::Cow;

use crate::backend::{ModuleSource, wgsl};
use crate::{
    AutoRepresentation, AutoRepresentationRef, CompilerInfo, WgpuCompiledKernel, WgpuCompiler,
};
use cubecl_core::{ExecutionMode, WgpuCompilationOptions, prelude::*};
use cubecl_server::compiler::CompilationError;
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderStages,
};

#[cfg(feature = "spirv")]
use crate::backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use crate::backend::metal;
use cubecl_ir::DeviceProperties;
use cubecl_server::compiler::{ArtifactCompiler, ArtifactId, CompilationRecording};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_units};

use super::pipelines::MetadataLayout;

/// Compiles kernels for one wgpu device with `C`, the shader language that
/// device was set up for, and has the device build their pipelines.
///
/// The device is shared by every thread compiling at once: wgpu's device is
/// `Sync`, and the error scopes [`validated`](Self::validated) reads a
/// creation's refusal from are per thread.
#[derive(Debug)]
pub(crate) struct WgpuArtifactCompiler<C> {
    device: wgpu::Device,
    properties: DeviceProperties,
    options: WgpuCompilationOptions,
    backend: wgpu::Backend,
    _compiler: PhantomData<fn() -> C>,
}

impl<C> WgpuArtifactCompiler<C> {
    pub(crate) fn new(
        device: wgpu::Device,
        properties: DeviceProperties,
        options: WgpuCompilationOptions,
        backend: wgpu::Backend,
    ) -> Self {
        Self {
            device,
            properties,
            options,
            backend,
            _compiler: PhantomData,
        }
    }
}

/// A compiled shader and what building and launching its pipeline needs.
pub struct WgpuArtifact {
    pub entrypoint_name: String,
    /// The shader's text: what a module is built from unless `repr` says
    /// the device takes something else.
    pub source: String,
    /// The tag of the language `source` is written in.
    pub lang: &'static str,
    pub repr: Option<AutoRepresentation>,
    pub compiler_info: CompilerInfo,
    pub io: Option<Arc<[BufferIOAttr]>>,
    /// The pipeline finalizing built from it; `None` for one read from the
    /// store, which is built when it is loaded.
    pub pipeline: Option<WgpuCompiledKernel>,
}

impl<C: WgpuCompiler> ArtifactCompiler for WgpuArtifactCompiler<C> {
    type Variant = MetadataLayout;
    type Lowered = CompiledKernel<C>;
    type Artifact = WgpuArtifact;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<MetadataLayout>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        let mut compiler = C::init(self.backend, &self.options);
        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            compiler.compile_kernel(kernel, definition, &self.options)
        })?;

        if logger.compilation_source_activated() {
            lowered.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                id.kernel.clone(),
            ));
        }
        logger.log_compilation(&lowered);

        compiler.validate_ir(&lowered.repr, &self.properties)?;

        Ok(lowered)
    }

    /// Has the device build the pipeline too: naga and the driver compile
    /// there, which is most of what a wgpu kernel costs.
    fn finalize(
        &self,
        id: &ArtifactId<MetadataLayout>,
        mut lowered: Self::Lowered,
    ) -> Result<WgpuArtifact, LaunchError> {
        let compiler = C::init(self.backend, &self.options);
        // The compiled kernel's per-buffer answer, before the repr is
        // consumed: what the write scope stages from.
        let io = lowered.io.take().map(Arc::from);
        let (compiler_info, repr) = compiler.normalize_repr(lowered.repr);

        let mut artifact = WgpuArtifact {
            entrypoint_name: lowered.entrypoint_name,
            source: lowered.source,
            lang: compiler.lang_tag(),
            repr,
            compiler_info,
            io,
            pipeline: None,
        };
        artifact.pipeline = Some(self.build(id, &artifact)?);
        Ok(artifact)
    }
}

impl<C> WgpuArtifactCompiler<C> {
    /// Has the device build `artifact`'s module and pipeline, laid out for
    /// `id`'s metadata.
    pub fn build(
        &self,
        id: &ArtifactId<MetadataLayout>,
        artifact: &WgpuArtifact,
    ) -> Result<WgpuCompiledKernel, CompilationError> {
        let repr = artifact.repr.as_ref().map(|repr| repr.as_ref());
        let module = self.create_module(
            &artifact.entrypoint_name,
            id.kernel.cube_dim.into(),
            ModuleSource::resolve(repr, artifact.lang, &artifact.source)?,
            id.kernel.mode,
        )?;
        let pipeline = self.create_pipeline(&artifact.entrypoint_name, repr, module, id.variant)?;
        Ok(WgpuCompiledKernel {
            pipeline,
            compiler_info: artifact.compiler_info,
            io: artifact.io.clone(),
        })
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

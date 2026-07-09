use std::fmt::Display;

use cubecl_core::{
    Compiler, WgpuCompilationOptions,
    backtrace::BackTrace,
    prelude::{CompiledKernel, KernelDefinition},
    server::{ComputeServer, LaunchError, ResourceLimitError},
};
#[cfg(feature = "msl")]
use cubecl_cpp::shared::MslComputeKernel;
use cubecl_ir::DeviceProperties;
use cubecl_runtime::compiler::CompilationError;
use derive_more::derive::From;

#[cfg(feature = "spirv")]
use crate::ParamsTransfer;
use crate::{CompilerInfo, WgpuServer};

use super::wgsl;

#[cfg(feature = "msl")]
pub use cubecl_cpp::MslCompiler;
#[cfg(feature = "spirv")]
pub use cubecl_spirv::SpirvCompiler;
pub use wgsl::WgslCompiler;

/// Compiler that dispatches to the most appropriate shader language backend for the active
/// `wgpu` backend.
///
/// The variant is selected at runtime by [`WgpuCompiler::init`] based on the `wgpu::Backend`
/// in use and the enabled cargo features (`spirv`, `msl`).
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum AutoCompiler {
    /// WGSL backend, available on every `wgpu` backend.
    Wgsl(WgslCompiler),
    /// SPIR-V backend used on Vulkan when the device supports the required features.
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvCompiler),
    /// Metal Shading Language backend used on the Metal backend.
    #[cfg(feature = "msl")]
    Msl(MslCompiler),
}

/// Owned compiled kernel representation matching the variants of [`AutoCompiler`].
#[derive(From)]
#[allow(clippy::large_enum_variant)]
pub enum AutoRepresentation {
    /// WGSL compute shader source.
    Wgsl(wgsl::ComputeShader),
    /// Compiled SPIR-V kernel.
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvKernel),
    /// Compiled Metal Shading Language kernel.
    #[cfg(feature = "msl")]
    Msl(MslComputeKernel),
}

/// Borrowed counterpart of [`AutoRepresentation`], useful when only read access is needed.
#[derive(From, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum AutoRepresentationRef<'a> {
    /// Borrowed WGSL compute shader.
    Wgsl(&'a wgsl::ComputeShader),
    /// Borrowed SPIR-V kernel.
    #[cfg(feature = "spirv")]
    SpirV(&'a cubecl_spirv::SpirvKernel),
    /// Borrowed Metal Shading Language kernel.
    #[cfg(feature = "msl")]
    Msl(&'a MslComputeKernel),
}

#[cfg(feature = "spirv")]
impl AutoRepresentation {
    /// Returns the SPIR-V kernel if this representation is the SPIR-V variant.
    pub fn as_spirv(&self) -> Option<&cubecl_spirv::SpirvKernel> {
        match self {
            AutoRepresentation::SpirV(repr) => Some(repr),
            _ => None,
        }
    }
}

#[cfg(feature = "msl")]
impl AutoRepresentation {
    /// Returns the MSL kernel if this representation is the MSL variant.
    pub fn as_msl(&self) -> Option<&MslComputeKernel> {
        match self {
            AutoRepresentation::Msl(repr) => Some(repr),
            _ => None,
        }
    }
}

impl AutoRepresentation {
    /// Borrow this representation as an [`AutoRepresentationRef`].
    pub fn as_ref(&self) -> AutoRepresentationRef<'_> {
        match self {
            AutoRepresentation::Wgsl(compute_shader) => AutoRepresentationRef::Wgsl(compute_shader),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(spirv_kernel) => AutoRepresentationRef::SpirV(spirv_kernel),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(compute_shader) => AutoRepresentationRef::Msl(compute_shader),
        }
    }
}

impl Display for AutoRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoRepresentation::Wgsl(compute_shader) => compute_shader.fmt(f),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(spirv_kernel) => spirv_kernel.fmt(f),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(compute_shader) => compute_shader.fmt(f),
        }
    }
}

impl Compiler for AutoCompiler {
    type Representation = AutoRepresentation;

    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        let kernel = match self {
            AutoCompiler::Wgsl(wgsl_compiler) => {
                Compiler::compile(wgsl_compiler, kernel, compilation_options)?.into()
            }
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(spirv_compiler) => {
                Compiler::compile(spirv_compiler, kernel, compilation_options)?.into()
            }
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(msl_compiler) => {
                // override compilation options with cpp compiler options for metal
                use cubecl_cpp;
                let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
                Compiler::compile(msl_compiler, kernel, &compilation_options)?.into()
            }
        };

        Ok(kernel)
    }

    fn extension(&self) -> &'static str {
        match self {
            AutoCompiler::Wgsl(_) => "wgsl",
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => "spv",
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(_) => "msl",
        }
    }
}

impl WgpuCompiler for AutoCompiler {
    fn init(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> Self {
        let _ = options; // Unused without `spirv` feature
        match backend {
            #[cfg(feature = "spirv")]
            wgpu::Backend::Vulkan if options.supports_vulkan_compiler => {
                AutoCompiler::SpirV(Default::default())
            }
            #[cfg(feature = "msl")]
            wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
            _ => AutoCompiler::Wgsl(Default::default()),
        }
    }

    fn compile_kernel(
        &mut self,
        server: &mut WgpuServer<AutoCompiler>,
        kernel: <WgpuServer<AutoCompiler> as ComputeServer>::Kernel,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        match self {
            AutoCompiler::Wgsl(_) => kernel.compile(self, &server.compilation_options),
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => {
                #[cfg(feature = "spirv-dump")]
                let (name, id) = (kernel.name().to_string(), kernel.id());
                let compiled = crate::vulkan::compile(self, server, kernel)?;
                #[cfg(feature = "spirv-dump")]
                if let Some(spirv) = compiled.repr.as_ref().and_then(|r| r.as_spirv()) {
                    crate::vulkan::dump_spirv(spirv, &name, id);
                }
                Ok(compiled)
            }
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(_) => kernel.compile(self, &server.compilation_options),
        }
    }

    fn lang_tag(&self) -> &'static str {
        match self {
            AutoCompiler::Wgsl(_) => "wgsl",
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => "spirv",
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(_) => "msl",
        }
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_size,
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size,
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        check_shared_memory(shared_bytes, props)
    }

    fn normalize_repr(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>) {
        let compiler_info = match &repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentation::SpirV(repr)) => CompilerInfo::Vulkan {
                params_transfer: match repr.immediate_size {
                    Some(_) => ParamsTransfer::Immediate,
                    None => ParamsTransfer::Uniform,
                },
            },
            #[cfg(feature = "msl")]
            Some(AutoRepresentation::Msl(_)) => CompilerInfo::Metal,
            Some(AutoRepresentation::Wgsl(_)) => CompilerInfo::WGSL,
            None => CompilerInfo::None,
        };

        (compiler_info, repr)
    }
}

impl WgpuCompiler for WgslCompiler {
    fn init(_backend: wgpu::Backend, _options: &WgpuCompilationOptions) -> Self {
        Self
    }

    fn compile_kernel(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        kernel.compile(self, &server.compilation_options)
    }

    fn lang_tag(&self) -> &'static str {
        "wgsl"
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| repr.shared_memory_size);
        check_shared_memory(shared_bytes, props)
    }

    fn normalize_repr(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>) {
        (CompilerInfo::WGSL, repr.map(|r| r.into()))
    }
}

#[cfg(feature = "msl")]
impl WgpuCompiler for MslCompiler {
    fn init(_backend: wgpu::Backend, _options: &WgpuCompilationOptions) -> Self {
        Self::default()
    }

    fn compile_kernel(
        &mut self,
        _server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        // The MSL compiler uses its own CompilationOptions, not WgpuCompilationOptions.
        let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        kernel.compile(self, &compilation_options)
    }

    fn lang_tag(&self) -> &'static str {
        "msl"
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| repr.shared_memory_size);
        check_shared_memory(shared_bytes, props)
    }

    fn normalize_repr(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>) {
        (CompilerInfo::Metal, repr.map(|r| r.into()))
    }
}

#[cfg(feature = "spirv")]
impl WgpuCompiler for cubecl_spirv::SpirvCompiler {
    fn init(_backend: wgpu::Backend, _options: &WgpuCompilationOptions) -> Self {
        Self
    }

    fn compile_kernel(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        #[cfg(feature = "spirv-dump")]
        let (name, id) = (kernel.name().to_string(), kernel.id());
        let compiled = crate::vulkan::compile(self, server, kernel)?;
        #[cfg(feature = "spirv-dump")]
        if let Some(spirv) = compiled.repr.as_ref() {
            crate::vulkan::dump_spirv(spirv, &name, id);
        }
        Ok(compiled)
    }

    fn lang_tag(&self) -> &'static str {
        "spirv"
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| repr.shared_size);
        check_shared_memory(shared_bytes, props)
    }

    fn normalize_repr(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>) {
        let params_transfer = match repr.as_ref().and_then(|r| r.immediate_size) {
            Some(_) => ParamsTransfer::Immediate,
            None => ParamsTransfer::Uniform,
        };
        (
            CompilerInfo::Vulkan { params_transfer },
            repr.map(|r| r.into()),
        )
    }
}

fn check_shared_memory(
    shared_bytes: Option<usize>,
    props: &DeviceProperties,
) -> Result<(), LaunchError> {
    let max_smem = props.hardware.max_shared_memory_size;
    if let Some(shared_bytes) = shared_bytes
        && shared_bytes > max_smem
    {
        return Err(ResourceLimitError::SharedMemory {
            requested: shared_bytes,
            max: max_smem,
            backtrace: BackTrace::capture(),
        }
        .into());
    }
    Ok(())
}

/// Extension trait implemented by every compiler usable with the `wgpu` runtime.
///
/// The base [`Compiler`] trait already exposes a `compile` method that turns a
/// [`KernelDefinition`] into a backend representation. [`WgpuCompiler`] sits one level
/// higher: it owns the wgpu-specific lifecycle around a [`CubeTask`](cubecl_runtime::compiler::CubeTask)
/// kernel — initializing the compiler for a given `wgpu::Backend`, compiling a kernel using
/// the server's [`WgpuCompilationOptions`], validating the resulting IR against the device,
/// and projecting the typed representation into the runtime-erased [`AutoRepresentation`].
pub trait WgpuCompiler: Compiler {
    /// Build the compiler instance appropriate for the given `wgpu` backend.
    ///
    /// `options` is consulted to decide between alternative implementations (for example, to
    /// opt into the SPIR-V compiler on Vulkan when the device advertises the required
    /// features).
    fn init(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> Self;

    /// Validate that the compiled representation fits within the device's resource limits.
    ///
    /// Today this checks shared memory usage; additional checks may be added without
    /// breaking the contract.
    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError>;

    /// Compile a runtime kernel into a [`CompiledKernel`] ready for pipeline creation.
    ///
    /// Distinct from [`Compiler::compile`], which only translates a [`KernelDefinition`].
    /// This entry point operates on a full server-level kernel and pulls compilation
    /// options from `server`, so its signature cannot collide with the base trait method.
    fn compile_kernel(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
    ) -> Result<CompiledKernel<Self>, CompilationError>;

    /// Short identifier of the shader language produced by this compiler (e.g. `"wgsl"`).
    ///
    /// Used for logging and debug-info tagging.
    fn lang_tag(&self) -> &'static str;

    /// Normalize the backend-specific representation into the [`AutoRepresentation`] shared
    /// by every wgpu compiler, and report the [`CompilerInfo`] derived from it.
    ///
    /// The [`CompilerInfo`] tells the server which parameter-passing strategy to use for
    /// the resulting pipeline.
    fn normalize_repr(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>);
}

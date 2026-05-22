use std::fmt::Display;

use cubecl_core::{
    Compiler, ExecutionMode, WgpuCompilationOptions,
    backtrace::BackTrace,
    ir::StorageType,
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

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum AutoCompiler {
    Wgsl(WgslCompiler),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvCompiler),
    #[cfg(feature = "msl")]
    Msl(MslCompiler),
}

#[derive(From)]
#[allow(clippy::large_enum_variant)]
pub enum AutoRepresentation {
    Wgsl(wgsl::ComputeShader),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvKernel),
    #[cfg(feature = "msl")]
    Msl(MslComputeKernel),
}

#[derive(From, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum AutoRepresentationRef<'a> {
    Wgsl(&'a wgsl::ComputeShader),
    #[cfg(feature = "spirv")]
    SpirV(&'a cubecl_spirv::SpirvKernel),
    #[cfg(feature = "msl")]
    Msl(&'a MslComputeKernel),
}

#[cfg(feature = "spirv")]
impl AutoRepresentation {
    pub fn as_spirv(&self) -> Option<&cubecl_spirv::SpirvKernel> {
        match self {
            AutoRepresentation::SpirV(repr) => Some(repr),
            _ => None,
        }
    }
}

#[cfg(feature = "msl")]
impl AutoRepresentation {
    pub fn as_msl(&self) -> Option<&MslComputeKernel> {
        match self {
            AutoRepresentation::Msl(repr) => Some(repr),
            _ => None,
        }
    }
}

impl AutoRepresentation {
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
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let kernel = match self {
            AutoCompiler::Wgsl(wgsl_compiler) => {
                Compiler::compile(wgsl_compiler, kernel, compilation_options, mode, addr_type)?
                    .into()
            }
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(spirv_compiler) => {
                Compiler::compile(spirv_compiler, kernel, compilation_options, mode, addr_type)?
                    .into()
            }
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(msl_compiler) => {
                // override compilation options with cpp compiler options for metal
                use cubecl_cpp;
                let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
                Compiler::compile(msl_compiler, kernel, &compilation_options, mode, addr_type)?
                    .into()
            }
        };

        Ok(kernel)
    }

    fn elem_size(&self, elem: cubecl_core::ir::ElemType) -> usize {
        match self {
            AutoCompiler::Wgsl(wgsl_compiler) => wgsl_compiler.elem_size(elem),
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(spirv_compiler) => spirv_compiler.elem_size(elem),
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(msl_compiler) => msl_compiler.elem_size(elem),
        }
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

    fn wgpu_compile(
        &mut self,
        server: &mut WgpuServer<AutoCompiler>,
        kernel: <WgpuServer<AutoCompiler> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        match self {
            AutoCompiler::Wgsl(_) => kernel.compile(
                self,
                &server.compilation_options,
                mode,
                kernel.address_type(),
            ),
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => {
                #[cfg(feature = "spirv-dump")]
                let (name, id) = (kernel.name().to_string(), kernel.id());
                let compiled = crate::vulkan::compile(self, server, kernel, mode)?;
                #[cfg(feature = "spirv-dump")]
                if let Some(spirv) = compiled.repr.as_ref().and_then(|r| r.as_spirv()) {
                    crate::vulkan::dump_spirv(spirv, &name, id);
                }
                Ok(compiled)
            }
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(_) => kernel.compile(
                self,
                &server.compilation_options,
                mode,
                kernel.address_type(),
            ),
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
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        check_shared_memory(shared_bytes, props)
    }

    fn to_auto(
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
        Self::default()
    }

    fn wgpu_compile(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        kernel.compile(
            self,
            &server.compilation_options,
            mode,
            kernel.address_type(),
        )
    }

    fn lang_tag(&self) -> &'static str {
        "wgsl"
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| repr.shared_memory_bytes());
        check_shared_memory(shared_bytes, props)
    }

    fn to_auto(
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

    fn wgpu_compile(
        &mut self,
        _server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        // The MSL compiler uses its own CompilationOptions, not WgpuCompilationOptions.
        let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        kernel.compile(self, &compilation_options, mode, kernel.address_type())
    }

    fn lang_tag(&self) -> &'static str {
        "msl"
    }

    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| repr.shared_memory_size());
        check_shared_memory(shared_bytes, props)
    }

    fn to_auto(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>) {
        (CompilerInfo::Metal, repr.map(|r| r.into()))
    }
}

#[cfg(feature = "spirv")]
impl<T: cubecl_spirv::SpirvTarget> WgpuCompiler for cubecl_spirv::SpirvCompiler<T> {
    fn init(_backend: wgpu::Backend, _options: &WgpuCompilationOptions) -> Self {
        Self::default()
    }

    fn wgpu_compile(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<Self>, CompilationError> {
        #[cfg(feature = "spirv-dump")]
        let (name, id) = (kernel.name().to_string(), kernel.id());
        let compiled = crate::vulkan::compile(self, server, kernel, mode)?;
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

    fn to_auto(
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

pub trait WgpuCompiler: Compiler {
    fn init(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> Self;
    fn validate_ir(
        &self,
        repr: &Option<Self::Representation>,
        props: &DeviceProperties,
    ) -> Result<(), LaunchError>;
    fn wgpu_compile(
        &mut self,
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<Self>, CompilationError>;
    fn lang_tag(&self) -> &'static str;
    fn to_auto(
        &self,
        repr: Option<Self::Representation>,
    ) -> (CompilerInfo, Option<AutoRepresentation>);
}

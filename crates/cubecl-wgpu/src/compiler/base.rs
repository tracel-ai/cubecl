use std::fmt::Display;

use cubecl_common::ExecutionMode;
use cubecl_core::{
    Compiler, WgpuCompilationOptions,
    prelude::{CompiledKernel, KernelDefinition},
    server::ComputeServer,
};
use derive_more::derive::From;

use crate::{WgpuServer, WgslCompiler};

use super::wgsl::ComputeShader;

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum AutoCompiler {
    Wgsl(WgslCompiler),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvCompiler),
}

#[derive(From)]
#[allow(clippy::large_enum_variant)]
pub enum AutoRepresentation {
    Wgsl(ComputeShader),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvKernel),
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

impl Display for AutoRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoRepresentation::Wgsl(compute_shader) => compute_shader.fmt(f),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(spirv_kernel) => spirv_kernel.fmt(f),
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
    ) -> Self::Representation {
        match self {
            AutoCompiler::Wgsl(wgsl_compiler) => {
                Compiler::compile(wgsl_compiler, kernel, compilation_options, mode).into()
            }
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(spirv_compiler) => {
                Compiler::compile(spirv_compiler, kernel, compilation_options, mode).into()
            }
        }
    }

    fn elem_size(&self, elem: cubecl_core::ir::Elem) -> usize {
        match self {
            AutoCompiler::Wgsl(wgsl_compiler) => wgsl_compiler.elem_size(elem),
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(spirv_compiler) => spirv_compiler.elem_size(elem),
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            AutoCompiler::Wgsl(_) => "wgsl",
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => "spv",
        }
    }
}

impl AutoCompiler {
    pub fn compile(
        &mut self,
        server: &mut WgpuServer,
        kernel: <WgpuServer as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<Self> {
        match self {
            AutoCompiler::Wgsl(_) => kernel.compile(self, &server.compilation_options, mode),
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => crate::vulkan::compile(self, server, kernel, mode),
        }
    }
}

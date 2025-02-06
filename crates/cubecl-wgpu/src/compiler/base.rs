use std::fmt::Display;

use cubecl_common::ExecutionMode;
use cubecl_core::{
    prelude::{CompiledKernel, KernelDefinition},
    server::ComputeServer,
    Compiler, CompilerRepresentation, WgpuCompilationOptions,
};
use derive_more::derive::From;

use crate::{WgpuServer, WgslCompiler};

use super::wgsl::ComputeShader;

#[derive(Debug, Clone)]
pub enum DynCompiler {
    Wgsl(WgslCompiler),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvCompiler),
}

#[derive(From)]
#[allow(clippy::large_enum_variant)]
pub enum DynRepresentation {
    Wgsl(ComputeShader),
    #[cfg(feature = "spirv")]
    SpirV(cubecl_spirv::SpirvKernel),
}

#[cfg(feature = "spirv")]
impl DynRepresentation {
    pub fn as_spirv(&self) -> Option<&cubecl_spirv::SpirvKernel> {
        match self {
            DynRepresentation::SpirV(repr) => Some(repr),
            _ => None,
        }
    }
}

impl Display for DynRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynRepresentation::Wgsl(compute_shader) => compute_shader.fmt(f),
            #[cfg(feature = "spirv")]
            DynRepresentation::SpirV(spirv_kernel) => spirv_kernel.fmt(f),
        }
    }
}

impl CompilerRepresentation for DynRepresentation {
    fn shared_memory_size(&self) -> usize {
        0
    }
}

impl Compiler for DynCompiler {
    type Representation = DynRepresentation;

    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        match self {
            DynCompiler::Wgsl(wgsl_compiler) => {
                Compiler::compile(wgsl_compiler, kernel, compilation_options, mode).into()
            }
            #[cfg(feature = "spirv")]
            DynCompiler::SpirV(spirv_compiler) => {
                Compiler::compile(spirv_compiler, kernel, compilation_options, mode).into()
            }
        }
    }

    fn elem_size(&self, elem: cubecl_core::ir::Elem) -> usize {
        match self {
            DynCompiler::Wgsl(wgsl_compiler) => wgsl_compiler.elem_size(elem),
            #[cfg(feature = "spirv")]
            DynCompiler::SpirV(spirv_compiler) => spirv_compiler.elem_size(elem),
        }
    }
}

impl DynCompiler {
    pub fn compile_dyn(
        &mut self,
        server: &mut WgpuServer,
        kernel: <WgpuServer as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<Self> {
        match self {
            DynCompiler::Wgsl(_) => {
                <WgslCompiler as WgpuCompiler>::compile(self, server, kernel, mode)
            }
            #[cfg(feature = "spirv")]
            DynCompiler::SpirV(_) => {
                <super::spirv::VkSpirvCompiler as WgpuCompiler>::compile(self, server, kernel, mode)
            }
        }
    }
}

pub trait WgpuCompiler: Compiler {
    fn compile(
        dyn_comp: &mut DynCompiler,
        server: &mut WgpuServer,
        kernel: <WgpuServer as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<DynCompiler>;
}

//! Selecting between the HIP C++ backend and the LLVM backend.

use cubecl_core::prelude::KernelDefinition;
use cubecl_cpp::shared::CompilationOptions;
#[cfg(feature = "cpp")]
use cubecl_cpp::{ComputeKernel, shared::CppCompiler, target::Hip};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use cubecl_runtime::kernel::BufferIOAttr;

#[derive(Clone, Debug)]
pub enum HipCompiler {
    #[cfg(feature = "cpp")]
    Cpp(CppCompiler<Hip>),
    #[cfg(not(feature = "cpp"))]
    Llvm(cubecl_llvm::PlironCompiler),
}

impl Default for HipCompiler {
    fn default() -> Self {
        #[cfg(feature = "cpp")]
        {
            HipCompiler::Cpp(CppCompiler::default())
        }
        #[cfg(not(feature = "cpp"))]
        {
            HipCompiler::Llvm(cubecl_llvm::PlironCompiler {
                target: cubecl_llvm::LlvmTarget::AmdGpu,
            })
        }
    }
}

/// Options for whichever backend is selected. One struct rather than an enum:
/// `arch` is a device property, filled in regardless of which backend reads it.
#[derive(Debug, Default, Clone)]
pub struct HipCompilationOptions {
    pub cpp: CompilationOptions,
    /// gfx name of the device, e.g. `"gfx1201"`.
    pub arch: String,
}

pub enum HipRepresentation {
    #[cfg(feature = "cpp")]
    Cpp(ComputeKernel),
    #[cfg(not(feature = "cpp"))]
    Llvm(cubecl_llvm::AmdGpuModule),
}

// `ComputeKernel` does not implement `Debug` (only `Display`, for its rendered
// source), so this is written by hand rather than derived.
impl core::fmt::Debug for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "cpp")]
            HipRepresentation::Cpp(_) => f.write_str("HipRepresentation::Cpp"),
            #[cfg(not(feature = "cpp"))]
            HipRepresentation::Llvm(module) => f
                .debug_tuple("HipRepresentation::Llvm")
                .field(module)
                .finish(),
        }
    }
}

impl HipRepresentation {
    /// Shared memory the launch must reserve. Both backends give their kernels one block of
    /// dynamic shared memory, so this is what the launch passes as `sharedMemBytes`.
    pub fn shared_memory_size(&self) -> usize {
        match self {
            #[cfg(feature = "cpp")]
            HipRepresentation::Cpp(kernel) => kernel.shared_memory_size,
            #[cfg(not(feature = "cpp"))]
            HipRepresentation::Llvm(module) => module.shared_memory_size,
        }
    }
}

impl core::fmt::Display for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "cpp")]
            HipRepresentation::Cpp(kernel) => write!(f, "{kernel}"),
            #[cfg(not(feature = "cpp"))]
            HipRepresentation::Llvm(module) => write!(f, "{}", module.ir),
        }
    }
}

impl Compiler for HipCompiler {
    type Representation = HipRepresentation;
    type CompilationOptions = HipCompilationOptions;

    /// Forwarded from the wrapped backend. The trait's default answers `None`,
    /// which reads as every buffer both read and written -- and that would make
    /// a pure output look like an input, so a relaunch repairing a tainted
    /// buffer would be skipped for reading what it only writes.
    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        match repr {
            #[cfg(feature = "cpp")]
            HipRepresentation::Cpp(kernel) => <CppCompiler<Hip> as Compiler>::buffer_io(kernel),
            #[cfg(not(feature = "cpp"))]
            HipRepresentation::Llvm(module) => Some(module.io.clone()),
        }
    }

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        match self {
            #[cfg(feature = "cpp")]
            HipCompiler::Cpp(compiler) => Ok(HipRepresentation::Cpp(
                compiler.compile(kernel, &options.cpp)?,
            )),
            #[cfg(not(feature = "cpp"))]
            HipCompiler::Llvm(compiler) => {
                let pliron_options = cubecl_llvm::PlironOptions {
                    arch: options.arch.clone(),
                };
                match compiler.compile(kernel, &pliron_options)? {
                    cubecl_llvm::PlironArtifact::AmdGpuCode(module) => {
                        Ok(HipRepresentation::Llvm(module))
                    }
                    cubecl_llvm::PlironArtifact::Jit(_) => {
                        unreachable!("the HIP runtime always configures LlvmTarget::AmdGpu")
                    }
                }
            }
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            #[cfg(feature = "cpp")]
            HipCompiler::Cpp(compiler) => compiler.extension(),
            #[cfg(not(feature = "cpp"))]
            HipCompiler::Llvm(_) => "ll",
        }
    }
}

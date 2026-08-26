//! Selecting between the HIP C++ backend and the LLVM backend.
//!
//! `Runtime::Compiler` is a single associated type, so supporting both means one
//! enum that dispatches internally rather than two runtimes. The choice is made
//! once per process from `CUBECL_HIP_COMPILER` and is deliberately not part of
//! the device configuration: it selects a whole toolchain, not a tuning knob.

use cubecl_core::prelude::KernelDefinition;
use cubecl_cpp::shared::{CompilationOptions, CppCompiler};
use cubecl_cpp::{ComputeKernel, target::Hip};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use cubecl_runtime::kernel::BufferIOAttr;

/// Environment variable selecting the backend. `llvm` picks the LLVM backend;
/// anything else, including unset, picks the C++ backend.
pub const COMPILER_ENV: &str = "CUBECL_HIP_COMPILER";

#[derive(Clone, Debug)]
pub enum HipCompiler {
    Cpp(CppCompiler<Hip>),
    #[cfg(feature = "llvm")]
    Llvm(cubecl_llvm::PlironCompiler),
}

impl Default for HipCompiler {
    fn default() -> Self {
        match std::env::var(COMPILER_ENV).as_deref() {
            #[cfg(feature = "llvm")]
            Ok("llvm") => HipCompiler::Llvm(cubecl_llvm::PlironCompiler {
                target: cubecl_llvm::LlvmTarget::AmdGpu,
            }),
            #[cfg(not(feature = "llvm"))]
            Ok("llvm") => {
                log::warn!(
                    "{COMPILER_ENV}=llvm ignored: cubecl-hip was built without the `llvm` feature"
                );
                HipCompiler::Cpp(CppCompiler::default())
            }
            _ => HipCompiler::Cpp(CppCompiler::default()),
        }
    }
}

/// Compilation options for whichever backend is selected. One struct rather than
/// an enum: `arch` is a property of the device and is filled in by the runtime
/// regardless of which backend consumes it.
#[derive(Debug, Default, Clone)]
pub struct HipCompilationOptions {
    pub cpp: CompilationOptions,
    /// gfx name of the device, e.g. `"gfx1201"`.
    pub arch: String,
}

pub enum HipRepresentation {
    Cpp(ComputeKernel),
    #[cfg(feature = "llvm")]
    Llvm(cubecl_llvm::AmdGpuModule),
}

// `ComputeKernel` does not implement `Debug` (only `Display`, for its rendered
// source), so this is written by hand rather than derived.
impl core::fmt::Debug for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HipRepresentation::Cpp(_) => f.write_str("HipRepresentation::Cpp"),
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(module) => {
                f.debug_tuple("HipRepresentation::Llvm").field(module).finish()
            }
        }
    }
}

impl HipRepresentation {
    /// Shared memory the launch must reserve. The LLVM backend does not support
    /// shared memory yet, so it always reports zero.
    pub fn shared_memory_size(&self) -> usize {
        match self {
            HipRepresentation::Cpp(kernel) => kernel.shared_memory_size,
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(_) => 0,
        }
    }
}

impl core::fmt::Display for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HipRepresentation::Cpp(kernel) => write!(f, "{kernel}"),
            #[cfg(feature = "llvm")]
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
            // The AMDGPU pipeline does not run `AnnotateGlobalVisibilityPass`,
            // so it has no stamped answer to forward.
            #[cfg(not(feature = "cpp"))]
            HipRepresentation::Llvm(_) => None,
        }
    }

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        match self {
            HipCompiler::Cpp(compiler) => Ok(HipRepresentation::Cpp(
                compiler.compile(kernel, &options.cpp)?,
            )),
            #[cfg(feature = "llvm")]
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
            HipCompiler::Cpp(compiler) => compiler.extension(),
            #[cfg(feature = "llvm")]
            HipCompiler::Llvm(_) => "ll",
        }
    }
}

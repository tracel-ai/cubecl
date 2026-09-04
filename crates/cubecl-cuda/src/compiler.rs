//! Selecting between the CUDA C++ backend and the LLVM backend.

use cubecl_core::ir::nvidia::SmArch;
use cubecl_core::prelude::KernelDefinition;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_cpp::{ComputeKernel, shared::CppCompiler, target::Cuda};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use cubecl_runtime::kernel::BufferIOAttr;

/// Which backend turns a `KernelDefinition` into something the CUDA driver can load.
///
/// Both are always compiled. The `cpp` feature selects which one a default [`CudaCompiler`]
/// runs, rather than which one exists, so a crate elsewhere in the graph enabling it changes
/// the default and cannot take the other away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaBackend {
    /// Transpile to CUDA C++ and hand the source to NVRTC.
    Cpp,
    /// Lower through pliron and LLVM to PTX.
    Llvm,
}

impl Default for CudaBackend {
    fn default() -> Self {
        if cfg!(feature = "cpp") {
            CudaBackend::Cpp
        } else {
            CudaBackend::Llvm
        }
    }
}

#[derive(Clone, Debug)]
pub enum CudaCompiler {
    Cpp(CppCompiler<Cuda>),
    Llvm(cubecl_llvm::PlironCompiler),
}

impl CudaCompiler {
    pub fn new(backend: CudaBackend) -> Self {
        match backend {
            CudaBackend::Cpp => CudaCompiler::Cpp(CppCompiler::default()),
            CudaBackend::Llvm => CudaCompiler::Llvm(cubecl_llvm::PlironCompiler {
                target: cubecl_llvm::LlvmTarget::Nvptx,
            }),
        }
    }
}

impl Default for CudaCompiler {
    fn default() -> Self {
        Self::new(CudaBackend::default())
    }
}

/// Options for whichever backend is selected. One struct rather than an enum: `arch` is a
/// device property, filled in regardless of which backend reads it.
#[derive(Debug, Default, Clone)]
pub struct CudaCompilationOptions {
    pub cpp: CompilationOptions,
    /// The device, as the runtime read its compute capability.
    pub arch: Option<SmArch>,
}

pub enum CudaRepresentation {
    Cpp(ComputeKernel),
    Llvm(cubecl_llvm::NvptxModule),
}

// `ComputeKernel` does not implement `Debug` (only `Display`, for its rendered source), so
// this is written by hand rather than derived.
impl core::fmt::Debug for CudaRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaRepresentation::Cpp(_) => f.write_str("CudaRepresentation::Cpp"),
            CudaRepresentation::Llvm(module) => f
                .debug_tuple("CudaRepresentation::Llvm")
                .field(module)
                .finish(),
        }
    }
}

impl CudaRepresentation {
    /// Shared memory the launch must reserve. Both backends give their kernels one block of
    /// dynamic shared memory, so this is what the launch passes as `sharedMemBytes`.
    pub fn shared_memory_size(&self) -> usize {
        match self {
            CudaRepresentation::Cpp(kernel) => kernel.shared_memory_size,
            CudaRepresentation::Llvm(module) => module.shared_memory_size,
        }
    }
}

impl core::fmt::Display for CudaRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaRepresentation::Cpp(kernel) => write!(f, "{kernel}"),
            CudaRepresentation::Llvm(module) => write!(f, "{}", module.ir),
        }
    }
}

impl Compiler for CudaCompiler {
    type Representation = CudaRepresentation;
    type CompilationOptions = CudaCompilationOptions;

    /// Forwarded from the wrapped backend. The trait's default answers `None`, which reads as
    /// every buffer both read and written -- and that would make a pure output look like an
    /// input, so a relaunch repairing a tainted buffer would be skipped for reading what it
    /// only writes.
    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        match repr {
            CudaRepresentation::Cpp(kernel) => <CppCompiler<Cuda> as Compiler>::buffer_io(kernel),
            CudaRepresentation::Llvm(module) => Some(module.io.clone()),
        }
    }

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        match self {
            CudaCompiler::Cpp(compiler) => Ok(CudaRepresentation::Cpp(
                compiler.compile(kernel, &options.cpp)?,
            )),
            CudaCompiler::Llvm(compiler) => {
                let pliron_options = cubecl_llvm::PlironOptions {
                    arch: None,
                    sm_arch: options.arch,
                    // Same flag the C++ backend reads, and it has to be: the server pushes the
                    // launch parameters from it, so both halves have to agree on the shape.
                    grid_constants: options.cpp.supports_features.grid_constants,
                };
                match compiler.compile(kernel, &pliron_options)? {
                    cubecl_llvm::PlironArtifact::NvptxCode(module) => {
                        Ok(CudaRepresentation::Llvm(module))
                    }
                    _ => unreachable!("the CUDA runtime always configures LlvmTarget::Nvptx"),
                }
            }
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            CudaCompiler::Cpp(compiler) => compiler.extension(),
            CudaCompiler::Llvm(_) => "ll",
        }
    }

    fn lang_tag(&self) -> &'static str {
        match self {
            CudaCompiler::Cpp(compiler) => compiler.lang_tag(),
            CudaCompiler::Llvm(compiler) => compiler.lang_tag(),
        }
    }
}

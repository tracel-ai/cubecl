//! The half of CUDA's compilation any thread can do: from a kernel to the
//! PTX `cuModuleLoadData` reads.

use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler, CudaRepresentation};
use crate::install::{cccl_include_path, include_path};
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording,
};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use std::ffi::{CStr, CString, c_char};
use std::str::FromStr;

/// Compiles kernels for one CUDA device, through whichever backend the build
/// selected: C++ through NVRTC, or LLVM straight to PTX.
#[derive(Debug)]
pub(crate) struct CudaArtifactCompiler {
    properties: DeviceProperties,
    options: CudaCompilationOptions,
    arch: CudaArchitecture,
}

impl CudaArtifactCompiler {
    pub(crate) fn new(
        properties: DeviceProperties,
        options: CudaCompilationOptions,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            properties,
            options,
            arch,
        }
    }

    /// The options kernels are compiled with.
    pub(crate) fn options(&self) -> &CudaCompilationOptions {
        &self.options
    }

    /// The architecture kernels are compiled for.
    pub(crate) fn arch(&self) -> &CudaArchitecture {
        &self.arch
    }
}

/// PTX ready for `cuModuleLoadData`, with what launching it needs: what the
/// compilation store keeps for a kernel.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct CudaArtifact {
    pub entrypoint_name: String,
    pub shared_mem_bytes: usize,
    pub ptx: Vec<std::ffi::c_char>,
    /// See [`CudaCompiledKernel::io`](super::modules::CudaCompiledKernel::io);
    /// defaulted for entries persisted before the field existed.
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
}

impl ArtifactCompiler for CudaArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<CudaCompiler>;
    type Artifact = CudaArtifact;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            CompiledKernel::compile(
                kernel,
                definition,
                &mut CudaCompiler::default(),
                &self.options,
            )
        })?;

        validate_shared_memory(
            &self.properties,
            lowered.repr.as_ref().map(|repr| repr.shared_memory_size()),
        )?;

        if logger.compilation_source_activated() {
            let extension = match lowered.repr {
                Some(CudaRepresentation::Llvm(_)) => "ll",
                _ => "cpp",
            };
            lowered.debug_info = Some(DebugInformation::new(extension, id.kernel.clone()));
            if extension == "cpp"
                && let Ok(formatted) = format_cpp(&lowered.source)
            {
                lowered.source = formatted;
            }
        }
        logger.log_compilation(&lowered);

        Ok(lowered)
    }

    /// The C++ source NVRTC compiles, which is slow enough that PTX already
    /// compiled from the same text is worth reusing. The LLVM backend's
    /// output is already PTX, so it has none.
    fn source(lowered: &Self::Lowered) -> Option<&str> {
        match backend_of(lowered) {
            CudaBackend::Cpp => Some(&lowered.source),
            CudaBackend::Llvm => None,
        }
    }

    fn finalize(
        &self,
        _id: &ArtifactId<Self::Variant>,
        mut lowered: Self::Lowered,
    ) -> Result<Self::Artifact, LaunchError> {
        let io = lowered.io.take();
        let (ptx, shared_mem_bytes) = match (backend_of(&lowered), &lowered.repr) {
            // What the LLVM backend hands back is already PTX, so no `nvrtc*`
            // call belongs here. The driver still JITs it when the module
            // loads, which is what it does with NVRTC's output too.
            (_, Some(CudaRepresentation::Llvm(module))) => {
                (module.ptx.clone(), module.shared_memory_size)
            }
            // A precompiled kernel has no representation to read the size
            // from: it declares its shared memory statically, so the launch
            // reserves none.
            (CudaBackend::Cpp, repr) => (
                self.compile_to_ptx(&lowered.source)?,
                repr.as_ref()
                    .map(|repr| repr.shared_memory_size())
                    .unwrap_or(0),
            ),
            (CudaBackend::Llvm, _) => {
                return Err(CompilationError::Generic {
                    reason: "the LLVM backend cannot load a precompiled kernel: it has no text to \
                         compile from"
                        .to_string(),
                    backtrace: BackTrace::capture(),
                }
                .into());
            }
        };

        Ok(CudaArtifact {
            entrypoint_name: lowered.entrypoint_name,
            shared_mem_bytes,
            ptx,
            io,
        })
    }
}

impl CudaArtifactCompiler {
    /// Compiles `source` to PTX with NVRTC.
    ///
    /// # Errors
    ///
    /// [`CompilationError::Generic`] carrying the compiler's own log, and the source that
    /// produced it, so a kernel the compiler refuses says why.
    fn compile_to_ptx(&self, source: &str) -> Result<Vec<c_char>, CompilationError> {
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        // SAFETY: Calling NVRTC FFI to create, compile, and extract PTX from a program.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let c_source = CString::from_str(source).unwrap();
            let program = cudarc::nvrtc::result::create_program(c_source.as_c_str(), None)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("{err}"),
                    backtrace: BackTrace::capture(),
                })?;
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            cudarc::nvrtc::result::get_ptx(program).map_err(|err| CompilationError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })
        }
    }
}

/// Which backend finalizes `lowered`: the one that compiled it, or, for a
/// precompiled kernel whose text passed the language check in
/// `CompiledKernel::compile`, the build's default — CUDA C++ goes through
/// NVRTC like a transpiled kernel, and the LLVM backend has no route for text.
fn backend_of(lowered: &CompiledKernel<CudaCompiler>) -> CudaBackend {
    match &lowered.repr {
        Some(CudaRepresentation::Cpp(_)) => CudaBackend::Cpp,
        Some(CudaRepresentation::Llvm(_)) => CudaBackend::Llvm,
        None => CudaBackend::default(),
    }
}

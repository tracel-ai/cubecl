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
use cubecl_server::kernel::{
    BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation, KernelParam,
};
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
    /// The parameter list of a precompiled binary, whose module `ptx` then
    /// holds instead of PTX. `None` for a kernel `CubeCL` compiled, which takes
    /// `CubeCL`'s own calling convention.
    #[serde(default)]
    pub params: Option<Vec<KernelParam>>,
    /// Whether `ptx` is a module compiled outside `CubeCL`, which no store
    /// keeps: its id is the caller's promise that it covers the image, and a
    /// rebuilt module under a stale id must not load from the previous run.
    #[serde(default)]
    pub precompiled: bool,
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
                Some(CudaRepresentation::Binary(_)) => "txt",
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
    /// output is already PTX, and a precompiled binary is already a module,
    /// so neither has one.
    fn source(lowered: &Self::Lowered) -> Option<&str> {
        if let Some(CudaRepresentation::Binary(_)) = lowered.repr {
            return None;
        }
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
        if let Some(CudaRepresentation::Binary(binary)) = lowered.repr {
            return Ok(CudaArtifact {
                entrypoint_name: binary.entrypoint_name,
                shared_mem_bytes: 0,
                ptx: module_image(&binary.image),
                io,
                params: Some(binary.params),
                precompiled: true,
            });
        }
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
            params: None,
            precompiled: false,
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

/// What `cuModuleLoadData` reads from `image`: a cubin or a fatbin as is,
/// and anything else as PTX text, which the driver reads up to a NUL. PTX
/// handed over without one gets one, so the driver never reads past the
/// image.
fn module_image(image: &[u8]) -> Vec<c_char> {
    let mut module: Vec<c_char> = bytemuck::cast_slice(image).to_vec();
    if is_ptx_text(&module) && module.last() != Some(&0) {
        module.push(0);
    }
    module
}

/// Whether `module` is PTX text rather than a cubin (an ELF object) or a
/// fatbin, told apart by their magic numbers.
pub(crate) fn is_ptx_text(module: &[c_char]) -> bool {
    const ELF: [u8; 4] = [0x7f, b'E', b'L', b'F'];
    const FATBIN: [u8; 4] = 0xBA55_ED50u32.to_le_bytes();
    let magic: &[u8] = bytemuck::cast_slice(module.get(..4).unwrap_or(module));
    magic != ELF && magic != FATBIN
}

/// Which backend finalizes `lowered`: the one that compiled it, or, for a
/// precompiled kernel whose text passed the language check in
/// `CompiledKernel::compile`, the build's default — CUDA C++ goes through
/// NVRTC like a transpiled kernel, and the LLVM backend has no route for text.
fn backend_of(lowered: &CompiledKernel<CudaCompiler>) -> CudaBackend {
    match &lowered.repr {
        Some(CudaRepresentation::Cpp(_)) => CudaBackend::Cpp,
        Some(CudaRepresentation::Llvm(_)) => CudaBackend::Llvm,
        Some(CudaRepresentation::Binary(_)) | None => CudaBackend::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::{is_ptx_text, module_image};
    use crate::compiler::CudaCompiler;
    use cubecl_environment::bytes::Bytes;
    use cubecl_server::compiler::Compiler;
    use cubecl_server::kernel::PrecompiledBinary;

    #[test]
    fn ptx_without_a_nul_gets_one() {
        let module = module_image(b".version 8.0");
        assert_eq!(module.last(), Some(&0));
        assert_eq!(module.len(), ".version 8.0".len() + 1);
    }

    #[test]
    fn ptx_with_a_nul_is_left_alone() {
        assert_eq!(
            module_image(b".version 8.0\0").len(),
            ".version 8.0".len() + 1
        );
    }

    #[test]
    fn cubins_and_fatbins_are_not_text() {
        let cubin = module_image(&[0x7f, b'E', b'L', b'F', 2, 1]);
        assert!(!is_ptx_text(&cubin));
        assert_eq!(cubin.len(), 6, "a cubin is loaded as is");

        let fatbin = module_image(&0xBA55_ED50u32.to_le_bytes());
        assert!(!is_ptx_text(&fatbin));
        assert_eq!(fatbin.len(), 4, "a fatbin is loaded as is");
    }

    fn load(image: Vec<u8>, entrypoint_name: &str) -> Result<(), String> {
        CudaCompiler::default()
            .load_binary(PrecompiledBinary {
                image: Bytes::from_bytes_vec(image),
                entrypoint_name: entrypoint_name.to_string(),
                params: Vec::new(),
            })
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    #[test]
    fn an_empty_image_is_refused() {
        let err = load(Vec::new(), "main").expect_err("nothing to load");
        assert!(err.contains("empty"), "says why: {err}");
    }

    #[test]
    fn an_entrypoint_with_a_nul_is_refused() {
        let err = load(b".version 8.0".to_vec(), "ma\0in").expect_err("not a C string");
        assert!(err.contains("NUL"), "says why: {err}");
    }
}

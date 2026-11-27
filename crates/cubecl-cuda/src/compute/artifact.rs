//! The half of CUDA's compilation any thread can do: from a kernel to the
//! PTX, and from the PTX to the machine code `cuModuleLoadData` reads.

use crate::compiler::{CudaBackend, CudaCompilationOptions, CudaCompiler, CudaRepresentation};
use crate::compute::events::driver_error;
use crate::install::{cccl_include_path, include_path};
use cubecl_core::{
    ir::{DeviceProperties, settings::DebugInfo},
    prelude::*,
};
use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_cpp::{formatter::format_cpp, shared::signature::source_forbids_contraction};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording, RegisterSpill,
};
use cubecl_server::driver::DriverError;
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use cudarc::driver::sys::{self, CUctx_st, CUjit_option, CUjitInputType, CUlinkState};
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::MaybeUninit;
use std::str::FromStr;

/// How much of the assembler's log a failed assembly reports.
const ASSEMBLER_LOG_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy)]
struct DriverContext(*mut CUctx_st);

// SAFETY: a CUDA context is a driver handle, not memory this process reads: any thread may make
// it current, and the driver API it is used through is thread-safe.
unsafe impl Send for DriverContext {}
// SAFETY: as for `Send`; sharing the handle shares nothing but the right to make it current.
unsafe impl Sync for DriverContext {}

/// Compiles kernels for one CUDA device, through whichever backend the build
/// selected: C++ through NVRTC, or LLVM straight to PTX.
#[derive(Debug)]
pub(crate) struct CudaArtifactCompiler {
    properties: DeviceProperties,
    options: CudaCompilationOptions,
    arch: CudaArchitecture,
    /// The CUDA context to assemble PTX on.
    context: DriverContext,
}

impl CudaArtifactCompiler {
    pub(crate) fn new(
        properties: DeviceProperties,
        options: CudaCompilationOptions,
        arch: CudaArchitecture,
        context: *mut CUctx_st,
    ) -> Self {
        Self {
            properties,
            options,
            arch,
            context: DriverContext(context),
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

/// A kernel ready for `cuModuleLoadData`, with what launching it needs: what the
/// compilation store keeps for a kernel.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct CudaArtifact {
    pub entrypoint_name: String,
    pub shared_mem_bytes: usize,
    pub ptx: Vec<std::ffi::c_char>,
    /// The machine code the driver assembled `ptx` to, which the load hands the driver.
    pub cubin: Vec<u8>,
    /// See [`CudaCompiledKernel::io`](super::modules::CudaCompiledKernel::io);
    /// defaulted for entries persisted before the field existed.
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
    /// What the assembler's log reported of the registers `cubin` spills; `None` when it
    /// reported nothing, and for entries persisted before the field existed.
    #[serde(default)]
    pub register_spill: Option<RegisterSpill>,
}

/// The machine code the driver assembled, and the registers its log reported it spills.
struct Assembled {
    cubin: Vec<u8>,
    register_spill: Option<RegisterSpill>,
}

/// The registers the assembler's verbose `info` log reports spilling, from its
/// `N bytes stack frame, N bytes spill stores, N bytes spill loads` line, summed over the
/// functions it lists. A 32-bit register a spill store moves is 4 bytes. `None` when no line
/// reports spill stores.
fn register_spill(info: &str) -> Option<RegisterSpill> {
    info.lines()
        .filter_map(|line| {
            let stores = bytes_before(line, "bytes spill stores")?;
            let stack = bytes_before(line, "bytes stack frame").unwrap_or(0);
            Some(RegisterSpill {
                spilled_registers: u32::try_from(stores.div_ceil(4)).unwrap_or(u32::MAX),
                scratch_bytes: stack,
            })
        })
        .reduce(|total, spill| RegisterSpill {
            spilled_registers: total
                .spilled_registers
                .saturating_add(spill.spilled_registers),
            scratch_bytes: total.scratch_bytes.max(spill.scratch_bytes),
        })
}

/// The number written right before `label` in `line`.
fn bytes_before(line: &str, label: &str) -> Option<u64> {
    let before = &line[..line.find(label)?];
    before.split_whitespace().next_back()?.parse().ok()
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
        id: &ArtifactId<Self::Variant>,
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

        let assembled = self.assemble(&ptx, id.kernel.debug_info() != DebugInfo::None)?;
        Ok(CudaArtifact {
            entrypoint_name: lowered.entrypoint_name,
            shared_mem_bytes,
            ptx,
            cubin: assembled.cubin,
            io,
            register_spill: assembled.register_spill,
        })
    }
}

impl CudaArtifactCompiler {
    /// Assembles `ptx` into the machine code of this context's device, with the line table of
    /// the PTX's `.loc` directives if `line_info` is set.
    ///
    /// # Errors
    ///
    /// The driver call that failed, with the assembler's log when it wrote one: PTX the
    /// assembler rejects, or a kernel past the device's limits.
    fn assemble(&self, ptx: &[c_char], line_info: bool) -> Result<Assembled, CompilationError> {
        let mut log = vec![0u8; ASSEMBLER_LOG_BYTES];
        let mut info = vec![0u8; ASSEMBLER_LOG_BYTES];
        let mut options = vec![
            CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CUjit_option::CU_JIT_INFO_LOG_BUFFER,
            CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CUjit_option::CU_JIT_LOG_VERBOSE,
        ];
        // The driver reads a size or flag option's value as the number itself, not through a
        // pointer.
        let mut values = vec![
            log.as_mut_ptr() as *mut c_void,
            log.len() as *mut c_void,
            info.as_mut_ptr() as *mut c_void,
            info.len() as *mut c_void,
            std::ptr::without_provenance_mut(1),
        ];
        if line_info {
            options.push(CUjit_option::CU_JIT_GENERATE_LINE_INFO);
            values.push(std::ptr::without_provenance_mut(1));
        }

        // SAFETY: the context is the device's, valid for the server's life, and any thread may
        // make it current. `log`, `options` and `values` outlive the link state, which reads the
        // options and writes the log and how much of it it filled.
        let assembled = unsafe {
            cudarc::driver::result::ctx::set_current(self.context.0)
                .map_err(|err| driver_error("cuCtxSetCurrent", err))?;
            let mut state = MaybeUninit::uninit();
            sys::cuLinkCreate_v2(
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
                state.as_mut_ptr(),
            )
            .result()
            .map_err(|err| driver_error("cuLinkCreate", err))?;
            let state = state.assume_init();
            let assembled = Self::link(state, ptx);
            let _ = sys::cuLinkDestroy(state);
            assembled
        };

        let cubin = assembled.map_err(|err| match CompilationError::from(err) {
            CompilationError::Generic { reason, backtrace } => {
                let log = CStr::from_bytes_until_nul(&log).map_or("", |log| {
                    log.to_str().unwrap_or("the assembler's log is not UTF-8")
                });
                CompilationError::Generic {
                    reason: format!("[Assembly Error] {reason}\n{log}"),
                    backtrace,
                }
            }
            poisoned => poisoned,
        })?;
        let register_spill = CStr::from_bytes_until_nul(&info)
            .ok()
            .and_then(|info| info.to_str().ok())
            .and_then(register_spill);

        Ok(Assembled {
            cubin,
            register_spill,
        })
    }

    /// Assembles `ptx` with the link `state`, and copies out the machine code it yields.
    ///
    /// # Safety
    ///
    /// `state` is a live link state, and `ptx` the null-terminated PTX `cuModuleLoadData`
    /// would read.
    unsafe fn link(state: CUlinkState, ptx: &[c_char]) -> Result<Vec<u8>, DriverError> {
        // SAFETY: as the caller promises. The machine code is the link state's, and copied out
        // before the caller destroys the state, which frees it.
        unsafe {
            sys::cuLinkAddData_v2(
                state,
                CUjitInputType::CU_JIT_INPUT_PTX,
                ptx.as_ptr() as *mut c_void,
                ptx.len(),
                c"kernel".as_ptr(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
            .result()
            .map_err(|err| driver_error("cuLinkAddData", err))?;
            let mut cubin = core::ptr::null_mut();
            let mut size = 0usize;
            sys::cuLinkComplete(state, &mut cubin, &mut size)
                .result()
                .map_err(|err| driver_error("cuLinkComplete", err))?;
            Ok(core::slice::from_raw_parts(cubin as *const u8, size).to_vec())
        }
    }

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
        // NVRTC fuses multiply-adds by default, whatever the source's pragma says.
        let contraction = source_forbids_contraction(source).then_some("--fmad=false");
        options.extend(contraction);

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
                // On newer minor architectures (e.g. B300 sm_103a running on a CUDA 12.8 NVRTC
                // that only knows sm_100a), retry once with the major architecture target.
                if self.arch.version > 90 && !self.arch.version.is_multiple_of(10) {
                    let major_arch =
                        format!("--gpu-architecture=sm_{}a", (self.arch.version / 10) * 10);
                    let mut fallback_opts =
                        vec![major_arch.as_str(), include_option.as_str(), "-lineinfo"];
                    if cccl_include_path.exists() {
                        fallback_opts.push(&cccl_include_option);
                    }
                    fallback_opts.extend(contraction);
                    let fallback_prog =
                        cudarc::nvrtc::result::create_program(c_source.as_c_str(), None).map_err(
                            |err| CompilationError::Generic {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            },
                        )?;
                    if cudarc::nvrtc::result::compile_program(fallback_prog, &fallback_opts).is_ok()
                    {
                        let _ = cudarc::nvrtc::result::destroy_program(program);
                        return cudarc::nvrtc::result::get_ptx(fallback_prog).map_err(|err| {
                            CompilationError::Generic {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            }
                        });
                    }
                    let _ = cudarc::nvrtc::result::destroy_program(fallback_prog);
                }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spill_stores_count_the_registers_spilled() {
        let info = "ptxas info    : Function properties for kernel\n    \
                    1224 bytes stack frame, 1188 bytes spill stores, 1188 bytes spill loads\n\
                    ptxas info    : Used 255 registers, 380 bytes cmem[0]";

        assert_eq!(
            register_spill(info),
            Some(RegisterSpill {
                spilled_registers: 297,
                scratch_bytes: 1224,
            })
        );
    }

    #[test]
    fn a_log_without_spill_stores_reports_nothing() {
        assert_eq!(register_spill("ptxas info    : Used 32 registers"), None);
    }
}

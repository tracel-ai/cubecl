//! The half of HIP's compilation any thread can do: from a kernel to the
//! code object `hipModuleLoadData` reads.

use crate::compiler::{HipBackend, HipCompilationOptions, HipCompiler, HipRepresentation};
use crate::compute::status::checked;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_cpp::formatter::format_cpp;
use cubecl_environment::backtrace::BackTrace;
use cubecl_hip_sys::get_hip_include_path;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording,
};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};

/// Compiles kernels for one HIP device, through whichever backend the build
/// selected: C++ through HIP RTC, or LLVM straight to a linked code object.
#[derive(Debug)]
pub(crate) struct HipArtifactCompiler {
    properties: DeviceProperties,
    options: HipCompilationOptions,
}

impl HipArtifactCompiler {
    pub(crate) fn new(properties: DeviceProperties, options: HipCompilationOptions) -> Self {
        Self {
            properties,
            options,
        }
    }
}

/// A code object ready for `hipModuleLoadData`, with what launching it needs:
/// what the compilation store keeps for a kernel.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct HipArtifact {
    pub entrypoint_name: String,
    pub shared_mem_bytes: usize,
    pub binary: Vec<i8>,
    /// See [`HipCompiledKernel::io`](super::modules::HipCompiledKernel::io);
    /// defaulted for entries persisted before the field existed.
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
}

impl ArtifactCompiler for HipArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<HipCompiler>;
    type Artifact = HipArtifact;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            CompiledKernel::compile(
                kernel,
                definition,
                &mut HipCompiler::default(),
                &self.options,
            )
        })?;

        validate_shared_memory(
            &self.properties,
            lowered.repr.as_ref().map(|repr| repr.shared_memory_size()),
        )?;

        if logger.compilation_source_activated() {
            let extension = match lowered.repr {
                Some(HipRepresentation::Llvm(_)) => "ll",
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

    /// The C++ source HIP RTC compiles, which is slow enough that a code
    /// object already compiled from the same text is worth reusing. The LLVM
    /// backend's output is already the code object, so it has none.
    fn source(lowered: &Self::Lowered) -> Option<&str> {
        match backend_of(lowered) {
            HipBackend::Cpp => Some(&lowered.source),
            HipBackend::Llvm => None,
        }
    }

    fn finalize(
        &self,
        _id: &ArtifactId<Self::Variant>,
        mut lowered: Self::Lowered,
    ) -> Result<Self::Artifact, LaunchError> {
        let io = lowered.io.take();
        let (binary, shared_mem_bytes) = match (backend_of(&lowered), &lowered.repr) {
            // What the LLVM backend hands back is already a linked `ET_DYN`
            // code object, so no `hiprtc*` call belongs here.
            (_, Some(HipRepresentation::Llvm(module))) => {
                (to_signed(&module.code_object), module.shared_memory_size)
            }
            // A precompiled kernel has no representation to read the size
            // from: it declares its shared memory statically, so the launch
            // reserves none.
            (HipBackend::Cpp, repr) => (
                compile_to_binary(&lowered.source)?,
                repr.as_ref()
                    .map(|repr| repr.shared_memory_size())
                    .unwrap_or(0),
            ),
            (HipBackend::Llvm, _) => {
                return Err(CompilationError::Generic {
                    reason: "the LLVM backend cannot load a precompiled kernel: it has no text to compile from"
                        .to_string(),
                    backtrace: BackTrace::capture(),
                }
                .into());
            }
        };

        Ok(HipArtifact {
            entrypoint_name: lowered.entrypoint_name,
            shared_mem_bytes,
            binary,
            io,
        })
    }
}

/// Which backend finalizes `lowered`: the one that compiled it, or, for a
/// precompiled kernel whose text passed the language check in
/// `CompiledKernel::compile`, the build's default — HIP C++ goes through HIP
/// RTC like a transpiled kernel, and the LLVM backend has no route for text.
fn backend_of(lowered: &CompiledKernel<HipCompiler>) -> HipBackend {
    match &lowered.repr {
        Some(HipRepresentation::Cpp(_)) => HipBackend::Cpp,
        Some(HipRepresentation::Llvm(_)) => HipBackend::Llvm,
        None => HipBackend::default(),
    }
}

/// A HIP RTC program, destroyed on drop.
///
/// The handle owns the source, the compilation log and the compiled code
/// inside the RTC runtime, none of which the caller needs once the binary has
/// been copied out. A guard rather than a call at the end because every step
/// of the compilation below returns early on failure, and each one of those
/// paths used to leak the program.
struct RtcProgram(cubecl_hip_sys::hiprtcProgram);

impl Drop for RtcProgram {
    fn drop(&mut self) {
        // SAFETY: created by `hiprtcCreateProgram` below and destroyed exactly
        // once here, after the compiled code has been copied out.
        unsafe {
            cubecl_hip_sys::hiprtcDestroyProgram(&mut self.0 as *mut _);
        }
    }
}

/// Compile `source` to a device binary with HIP RTC.
///
/// # Errors
///
/// [`CompilationError::Generic`] carrying the compiler's own log, and the
/// source that produced it, so a kernel the driver refuses says why.
fn compile_to_binary(source: &str) -> Result<Vec<i8>, CompilationError> {
    let source = CString::new(source).map_err(|err| CompilationError::Generic {
        reason: format!("The generated source is not a valid C string: {err}"),
        backtrace: BackTrace::capture(),
    })?;

    // SAFETY: `source` is null-terminated and outlives the call. The returned
    // handle is valid on success and owned by the guard from here on.
    let program = unsafe {
        let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();
        let status = cubecl_hip_sys::hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            std::ptr::null(), // program name seems unnecessary
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        checked("hiprtcCreateProgram", status)?;
        RtcProgram(program)
    };

    let include_path = get_hip_include_path().map_err(|err| CompilationError::Generic {
        reason: format!("Unable to locate the HIP headers to compile against: {err}"),
        backtrace: BackTrace::capture(),
    })?;
    let include_option =
        CString::new(format!("-I{include_path}")).map_err(|err| CompilationError::Generic {
            reason: format!("The HIP include path is not a valid C string: {err}"),
            backtrace: BackTrace::capture(),
        })?;
    // needed for rocWMMA extension to compile
    let cpp_std_option = c"--std=c++17";
    let optimization_level = c"-O3";
    let mut options = [
        cpp_std_option.as_ptr(),
        include_option.as_ptr(),
        optimization_level.as_ptr(),
    ];

    // SAFETY: `program.0` is the handle created above, and `options` holds
    // null-terminated pointers that outlive the call.
    let status = unsafe {
        cubecl_hip_sys::hiprtcCompileProgram(program.0, options.len() as i32, options.as_mut_ptr())
    };
    if checked("hiprtcCompileProgram", status).is_err() {
        return Err(CompilationError::Generic {
            reason: format!(
                "{}\n[Source]  \n{}",
                compilation_log(&program),
                source.to_string_lossy()
            ),
            backtrace: BackTrace::capture(),
        });
    }

    // SAFETY: `program.0` compiled successfully above, so it has code to
    // report the size of and to copy out into a buffer of exactly that size.
    unsafe {
        let mut code_size: usize = 0;
        let status = cubecl_hip_sys::hiprtcGetCodeSize(program.0, &mut code_size);
        checked("hiprtcGetCodeSize", status)?;
        let mut code = vec![0; code_size];
        let status = cubecl_hip_sys::hiprtcGetCode(program.0, code.as_mut_ptr());
        checked("hiprtcGetCode", status)?;
        Ok(code)
    }
}

/// The compiler's log for a program it refused, indented under a heading.
///
/// Reports why the log itself is missing rather than failing on it: this runs
/// on a path that already has an error to report, and losing that error to a
/// second one would leave the caller with nothing.
fn compilation_log(program: &RtcProgram) -> String {
    let mut message = "[Compilation Error] ".to_string();
    // SAFETY: `program.0` is a valid handle; the log buffer is sized by the
    // call that reports its length, and read back as a C string.
    let log = unsafe {
        let mut log_size: usize = 0;
        let status =
            cubecl_hip_sys::hiprtcGetProgramLogSize(program.0, &mut log_size as *mut usize);
        if let Err(err) = checked("hiprtcGetProgramLogSize", status) {
            return message + &format!("\n the log's length is unavailable: {err}");
        }
        if log_size == 0 {
            return message + "\n No compilation logs found!";
        }
        let mut log_buffer = vec![0; log_size];
        let status = cubecl_hip_sys::hiprtcGetProgramLog(program.0, log_buffer.as_mut_ptr());
        if let Err(err) = checked("hiprtcGetProgramLog", status) {
            return message + &format!("\n the log itself is unavailable: {err}");
        }
        CStr::from_ptr(log_buffer.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    for line in log.split('\n').filter(|line| !line.is_empty()) {
        message += format!("\n    {line}").as_str();
    }
    message
}

/// A code object as the `c_char` slice the cache and `hipModuleLoadData` are written in.
///
/// `i8` and `u8` have the same layout, so this is the copy out of the compiler's buffer and
/// nothing more.
fn to_signed(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|byte| *byte as i8).collect()
}

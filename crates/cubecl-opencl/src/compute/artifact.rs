//! The half of `OpenCL`'s compilation any thread can do: from a kernel to the `OpenCL` C, and the
//! program and kernel the driver builds from it.

use crate::OpenClCompiler;
use crate::compute::cl;
use cl3::info_type::InfoType;
use cubecl_core::server::LaunchError;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_cpp::opencl::OpenClCompilationOptions;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording,
};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use opencl_sys::{
    CL_KERNEL_PROGRAM, CL_PROGRAM_BINARIES, CL_PROGRAM_BUILD_LOG, cl_context, cl_device_id,
    cl_kernel, cl_program,
};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

/// The build options every kernel gets: `OpenCL` C 3.0.
const OPTIONS: &CStr = c"-cl-std=CL3.0";

/// A kernel object, released with the last clone.
#[derive(Debug)]
pub(crate) struct KernelHandle(cl_kernel);

impl KernelHandle {
    pub(crate) fn raw(&self) -> cl_kernel {
        self.0
    }
}

// SAFETY: `OpenCL` kernel objects are thread-safe handles; only the server thread sets their
// arguments and enqueues them, one launch at a time.
unsafe impl Send for KernelHandle {}
// SAFETY: as above.
unsafe impl Sync for KernelHandle {}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        // The reference this handle owns.
        cl::api().clReleaseKernel(self.0);
    }
}

/// A kernel ready to launch.
#[derive(Debug, Clone)]
pub(crate) struct OpenClKernel {
    pub kernel: Arc<KernelHandle>,
    pub cube_dim: CubeDim,
    pub io: Option<Arc<[BufferIOAttr]>>,
    /// After its buffers, the kernel takes the dynamic metadata (a buffer) when it has one, then the
    /// info struct (by value) when it has one.
    pub has_dynamic_meta: bool,
    pub has_info: bool,
}

/// `OpenCL` C and the program the driver built from it, with what launching it needs: what the
/// compilation store keeps for a kernel.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct OpenClCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    source: String,
    /// The device binary of the program built from `source` (empty where the driver gives none):
    /// loading it takes the driver milliseconds, building `source` again up to a second.
    #[serde(with = "serde_bytes")]
    binary: Vec<u8>,
    io: Option<Vec<BufferIOAttr>>,
    has_dynamic_meta: bool,
    has_info: bool,
}

/// `OpenCL` C, and the kernel finalizing built from it.
#[derive(Debug)]
pub(crate) struct OpenClArtifact {
    /// What the store keeps.
    pub entry: OpenClCacheEntry,
    /// `None` for a source read from the store, which is built when it is loaded.
    pub kernel: Option<OpenClKernel>,
}

/// Compiles kernels to `OpenCL` C for one device, and has the driver build them.
///
/// Shared by every thread compiling at once: `OpenCL` contexts, programs and kernel creation are
/// thread-safe (setting a kernel's arguments is not, and only the server thread does).
#[derive(Debug)]
pub(crate) struct OpenClArtifactCompiler {
    context: cl_context,
    device: cl_device_id,
    properties: DeviceProperties,
    options: OpenClCompilationOptions,
}

// SAFETY: the context and device are thread-safe `OpenCL` handles, the context retained by this
// value.
unsafe impl Send for OpenClArtifactCompiler {}
// SAFETY: as above; `&self` only creates programs and kernels.
unsafe impl Sync for OpenClArtifactCompiler {}

impl Drop for OpenClArtifactCompiler {
    fn drop(&mut self) {
        cl::api().clReleaseContext(self.context);
    }
}

impl OpenClArtifactCompiler {
    pub(crate) fn new(
        context: cl_context,
        device: cl_device_id,
        properties: DeviceProperties,
        options: OpenClCompilationOptions,
    ) -> Self {
        // A reference of the compiler's own: it may outlive the server on a compiling thread.
        cl::api().clRetainContext(context);
        Self {
            context,
            device,
            properties,
            options,
        }
    }

    /// Has the driver load `entry`'s program from its binary, or build it from its source where
    /// there is no binary or the driver refuses it, and takes its kernel.
    pub(crate) fn load(&self, entry: &OpenClCacheEntry) -> Result<OpenClKernel, CompilationError> {
        let cube_dim = entry.cube_dim.into();
        let kernel = match self.load_binary(&entry.binary, &entry.entrypoint_name) {
            Some(kernel) => kernel,
            None => self.build(&entry.source, &entry.entrypoint_name)?,
        };
        Ok(OpenClKernel {
            kernel: Arc::new(kernel),
            cube_dim,
            io: entry.io.clone().map(Arc::from),
            has_dynamic_meta: entry.has_dynamic_meta,
            has_info: entry.has_info,
        })
    }
}

impl ArtifactCompiler for OpenClArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<OpenClCompiler>;
    type Artifact = OpenClArtifact;

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
                &mut OpenClCompiler::default(),
                &self.options,
            )
        })?;

        if logger.compilation_source_activated() {
            lowered.debug_info = Some(DebugInformation::new("opencl", id.kernel.clone()));
        }
        logger.log_compilation(&lowered);

        // Checked before the driver builds the program: it would refuse the kernel there anyway,
        // with a build error instead of a resource limit error.
        validate_shared_memory(
            &self.properties,
            lowered.repr.as_ref().map(|repr| repr.shared_memory_size),
        )?;

        Ok(lowered)
    }

    /// Has the driver build the program and kernel too: most of what an `OpenCL` kernel costs.
    fn finalize(
        &self,
        _id: &ArtifactId<()>,
        mut lowered: Self::Lowered,
    ) -> Result<OpenClArtifact, LaunchError> {
        let repr = lowered
            .repr
            .as_ref()
            .ok_or_else(|| CompilationError::Generic {
                reason: "the OpenCL compiler returned no kernel representation".into(),
                backtrace: BackTrace::capture(),
            })?;
        let (has_dynamic_meta, has_info) = (repr.has_dynamic_meta, repr.has_info);
        let cube_dim = lowered.cube_dim;
        let mut entry = OpenClCacheEntry {
            io: lowered.io.take(),
            entrypoint_name: lowered.entrypoint_name,
            cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
            source: lowered.source,
            binary: Vec::new(),
            has_dynamic_meta,
            has_info,
        };
        let kernel = self.load(&entry)?;
        entry.binary = binary(&kernel.kernel).unwrap_or_default();
        Ok(OpenClArtifact {
            entry,
            kernel: Some(kernel),
        })
    }

    const PREPARES: bool = true;

    /// Loads a stored kernel's program, which the store keeps as its binary and source.
    fn prepare(&self, _id: &ArtifactId<()>, artifact: &mut OpenClArtifact) {
        if artifact.kernel.is_none() {
            artifact.kernel = self.load(&artifact.entry).ok();
        }
    }
}

impl OpenClArtifactCompiler {
    /// The kernel `entrypoint_name` of the program the driver loads from `binary`, if it takes it.
    fn load_binary(&self, binary: &[u8], entrypoint_name: &str) -> Option<KernelHandle> {
        if binary.is_empty() {
            return None;
        }
        // SAFETY: one binary, for the one device.
        let program = unsafe {
            cl3::program::create_program_with_binary(self.context, &[self.device], &[binary])
        }
        .ok()?;
        self.kernel(program, c"", entrypoint_name).ok()
    }

    /// Builds `source` for this compiler's device with [`OPTIONS`], and takes its kernel
    /// `entrypoint_name`. A kernel that cannot run its cube's units is refused at launch.
    fn build(&self, source: &str, entrypoint_name: &str) -> Result<KernelHandle, CompilationError> {
        cl3::program::create_program_with_source(self.context, &[source])
            .map_err(|code| format!("clCreateProgramWithSource failed ({code})"))
            .and_then(|program| self.kernel(program, OPTIONS, entrypoint_name))
            .map_err(|reason| CompilationError::Generic {
                reason: format!("{reason}\n{source}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// Builds `program` with `options` and takes its kernel `entrypoint_name`, releasing the
    /// program (the kernel holds its own reference).
    fn kernel(
        &self,
        program: cl_program,
        options: &CStr,
        entrypoint_name: &str,
    ) -> Result<KernelHandle, String> {
        let built =
            cl3::program::build_program(program, &[self.device], options, None, ptr::null_mut());
        let kernel = match built {
            Err(code) => {
                let log = cl3::program::get_program_build_data(
                    program,
                    self.device,
                    CL_PROGRAM_BUILD_LOG,
                )
                .map(|bytes| {
                    String::from_utf8_lossy(&bytes)
                        .trim_end_matches('\0')
                        .to_string()
                })
                .unwrap_or_default();
                Err(format!(
                    "OpenCL build of {entrypoint_name} failed ({code}):\n{log}"
                ))
            }
            Ok(()) => CString::new(entrypoint_name)
                .map_err(|_| format!("kernel name {entrypoint_name:?} holds a NUL"))
                .and_then(|name| {
                    cl3::kernel::create_kernel(program, &name)
                        .map(KernelHandle)
                        .map_err(|code| {
                            format!("clCreateKernel({entrypoint_name}) failed ({code})")
                        })
                }),
        };
        cl::api().clReleaseProgram(program);
        kernel
    }
}

/// The device binary of the program `kernel` was taken from.
fn binary(kernel: &KernelHandle) -> Option<Vec<u8>> {
    let InfoType::Ptr(program) = cl3::kernel::get_kernel_info(kernel.0, CL_KERNEL_PROGRAM).ok()?
    else {
        return None;
    };
    match cl3::program::get_program_info(program as cl_program, CL_PROGRAM_BINARIES).ok()? {
        InfoType::VecVecUchar(mut binaries) => binaries.pop().filter(|binary| !binary.is_empty()),
        _ => None,
    }
}

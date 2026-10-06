use crate::compiler::{CudaBackend, CudaCompilationOptions};
use crate::compute::artifact::CudaArtifactCompiler;
use crate::compute::events::{EventProfiler, driver_error, poisons_device};
use crate::compute::modules::CudaModules;
use crate::compute::stream::Stream;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_cpp::cuda::arch::CudaArchitecture;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{ArtifactId, CompilationTarget, KernelLoader};
use cubecl_server::kernel::{BufferIOAttr, CubeKernel};
use cubecl_server::logging::ServerLogger;
use cudarc::driver::DriverError;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUstream};
use std::os::raw::c_void;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    /// The stream collectives run on. Kept on the context so a relocation —
    /// which reaches the context, not the server — can wait on it.
    pub comm_stream: CUstream,
    /// The modules loaded on the device, and how to load another.
    kernels: KernelLoader<CudaModules>,
    pub profiler: EventProfiler,
}

impl CudaContext {
    /// `backend` is which one compiles here; see [`CudaModules::new`].
    ///
    /// An environment switch drops every loaded kernel, so the new
    /// environment's PTX store is filled rather than bypassed. The modules
    /// those kernels named stay resident: nothing calls `cuModuleUnload`, here
    /// or anywhere else in this context, and unloading one a stream still has
    /// queued work against would be unsound. A process that switches
    /// environments a handful of times at startup pays a bounded price; one
    /// that switches repeatedly grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn new(
        compilation_options: CudaCompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
        backend: CudaBackend,
        comm_stream: CUstream,
    ) -> Self {
        let compiler = CudaArtifactCompiler {
            properties,
            options: compilation_options,
            arch,
        };

        Self {
            context,
            comm_stream,
            kernels: KernelLoader::new(CudaModules::new(compiler, backend)),
            profiler: EventProfiler::default(),
        }
    }

    /// The options kernels are compiled with, which the launch path reads
    /// to pass arguments the way the compiled code expects them.
    pub fn compilation_options(&self) -> &CudaCompilationOptions {
        &self.kernels.target().compiler().options
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    /// Loads `kernel`, whose id is `kernel_id`, on the device, compiling it
    /// first when no store holds it. A kernel already loaded costs a map
    /// lookup.
    pub fn load_kernel(
        &mut self,
        kernel: &dyn CubeKernel,
        kernel_id: KernelId,
        logger: &ServerLogger,
    ) -> Result<(), LaunchError> {
        let id = ArtifactId {
            kernel: kernel_id,
            variant: (),
        };
        self.kernels.load(kernel, id, logger).map(|_| ())
    }

    /// Queues `kernel` to be compiled with others, by the next kernel
    /// loaded for a launch.
    pub fn queue_kernel(&mut self, kernel: Box<dyn CubeKernel>) {
        self.kernels.enqueue(kernel, ());
    }

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — `None` when the kernel is not loaded or predates the
    /// answer, which the launch path reads as every buffer both read and
    /// written.
    pub fn kernel_io(&mut self, kernel_id: &KernelId) -> Option<Arc<[BufferIOAttr]>> {
        self.kernels
            .get(&ArtifactId {
                kernel: kernel_id.clone(),
                variant: (),
            })
            .and_then(|kernel| kernel.io.clone())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        let kernel = self
            .kernels
            .get(&ArtifactId {
                kernel: kernel_id.clone(),
                variant: (),
            })
            .expect("loaded before the launch was enqueued");
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| launch_failed("cuFuncSetAttribute", err))?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                resources,
            )
            .map_err(|err| launch_failed("cuLaunchKernel", err))?;
        };

        Ok(())
    }
}

/// A refused launch error : maps to a poisoned-device error if that call happened right after
/// a fault.
fn launch_failed(op: &'static str, err: cudarc::driver::DriverError) -> LaunchError {
    match poisons_device(err.0) {
        true => driver_error(op, err).into(),
        false => LaunchError::Unknown {
            reason: format!("{err}"),
            backtrace: BackTrace::capture(),
        },
    }
}

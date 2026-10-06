//! The device context: its loaded kernels, and its open profiles.
//!
//! Compilation is memoized here rather than per stream, because a module is
//! loaded into the context and every stream sharing it can launch from the
//! same one. What a compiled kernel answers for beyond its entry point is
//! which of its bindings it writes, which is what a launch stages its write
//! scope from.

use super::storage::gpu::GpuResource;
use crate::compiler::{HipBackend, HipCompilationOptions};
use crate::compute::artifact::HipArtifactCompiler;
use crate::compute::events::EventProfiler;
use crate::compute::modules::HipModules;
use crate::compute::status::checked;
use crate::compute::stream::Stream;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::kernel::{BufferIOAttr, CubeKernel};
use cubecl_server::logging::ServerLogger;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    /// The modules loaded on the device, and how to load another.
    kernels: KernelLoader<HipModules>,
    pub profiler: EventProfiler,
}

impl HipContext {
    /// `fingerprint` and `backend` name where compiled kernels are stored; see
    /// [`HipModules::new`].
    ///
    /// An environment switch drops every loaded kernel, so the new
    /// environment's store is filled rather than bypassed. The modules those
    /// kernels named stay resident: nothing calls `hipModuleUnload`, here or
    /// anywhere else in this context, and unloading one a stream still has
    /// queued work against would be unsound. A process that switches
    /// environments a handful of times at startup pays a bounded price; one
    /// that switches repeatedly grows its resident modules without bound — see
    /// [`cubecl_environment::environment::activate`].
    pub fn new(
        compilation_options: HipCompilationOptions,
        properties: DeviceProperties,
        fingerprint: String,
        backend: HipBackend,
    ) -> Self {
        let compiler = HipArtifactCompiler {
            properties,
            options: compilation_options,
        };
        Self {
            kernels: KernelLoader::new(HipModules::new(compiler, fingerprint, backend)),
            profiler: EventProfiler::default(),
        }
    }

    /// Loads `kernel` on the device, compiling it first when no store holds
    /// it. A kernel already loaded costs a map lookup.
    pub fn load_kernel(
        &mut self,
        kernel: &dyn CubeKernel,
        logger: &ServerLogger,
    ) -> Result<(), LaunchError> {
        self.kernels.load(kernel, (), logger).map(|_| ())
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

    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self
            .kernels
            .get(&ArtifactId {
                kernel: kernel_id.clone(),
                variant: (),
            })
            .expect("loaded before the launch was enqueued");
        let cube_dim = kernel.cube_dim;

        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid HIP stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            let status = cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                cube_dim.x,
                cube_dim.y,
                cube_dim.z,
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            // Out of memory is told apart from the rest because the caller
            // can act on it — reclaim and relaunch — where nothing else here
            // is worth retrying.
            match checked("hipModuleLaunchKernel", status) {
                Ok(()) => Ok(()),
                Err(_) if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory => {
                    Err(LaunchError::OutOfMemory {
                        reason: format!("out of memory launching kernel {kernel_id:?}"),
                        backtrace: BackTrace::capture(),
                    })
                }
                Err(err) => Err(LaunchError::Unknown {
                    reason: format!("{err}, launching kernel {kernel_id:?}"),
                    backtrace: BackTrace::capture(),
                }),
            }
        }
    }
}

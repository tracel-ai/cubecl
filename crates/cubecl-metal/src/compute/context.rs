//! The device's compiled pipelines, behind the one type the server holds.

use crate::compute::artifact::MetalArtifactCompiler;
use crate::compute::pipelines::{MetalCompiledKernel, MetalPipelines};
use cubecl_core::ir::DeviceProperties;
use cubecl_core::prelude::KernelId;
use cubecl_core::server::LaunchError;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::kernel::CubeKernel;
use cubecl_server::logging::ServerLogger;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Compiles `CubeCL` IR to MSL and on to `MTLComputePipelineState`, caching results.
#[derive(Debug)]
pub struct MetalContext {
    kernels: KernelLoader<MetalPipelines>,
}

impl MetalContext {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        properties: DeviceProperties,
        compilation_options: cubecl_cpp::shared::CompilationOptions,
    ) -> Self {
        let compiler = MetalArtifactCompiler::new(device, properties, compilation_options);
        Self {
            kernels: KernelLoader::new(MetalPipelines::new(compiler)),
        }
    }

    /// The pipeline for `kernel`, whose id is `id`, compiling it first when
    /// no store holds it.
    pub fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        logger: &ServerLogger,
    ) -> Result<MetalCompiledKernel, LaunchError> {
        self.kernels.load(kernel, id, logger)
    }

    /// Compiles every queued kernel now, rather than inside the next launch.
    pub fn compile_queued(&mut self, logger: &ServerLogger) {
        self.kernels.compile_queued(logger);
    }

    /// Queues `kernel` to be compiled with others, by the next pipeline
    /// loaded for a launch.
    pub fn queue(&mut self, kernel: Box<dyn CubeKernel>) {
        self.kernels.enqueue(kernel, ());
    }
}

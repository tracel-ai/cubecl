//! The device's compiled pipelines, behind the one type the server holds.

use crate::compute::artifact::MetalArtifactCompiler;
use crate::compute::pipelines::{CompiledKernel, MetalPipelines};
use cubecl_core::ir::DeviceProperties;
use cubecl_core::server::LaunchError;
use cubecl_server::compiler::KernelLoader;
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
        let compiler = MetalArtifactCompiler {
            properties,
            options: compilation_options,
        };
        Self {
            kernels: KernelLoader::new(MetalPipelines::new(compiler, device)),
        }
    }

    /// The pipeline for `kernel`, compiling it first when no store holds it.
    pub fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        logger: &ServerLogger,
    ) -> Result<CompiledKernel, LaunchError> {
        self.kernels.load(kernel, (), logger).cloned()
    }
}

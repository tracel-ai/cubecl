//! How the CPU runtime compiles a kernel: through pliron and LLVM to a JIT
//! engine, specialized for the alignment of the buffers it is launched on.

use std::sync::Arc;

use cubecl_core::server::LaunchError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_llvm::{PlironCompiler, PlironOptions};
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording, CompilationTarget,
};
use cubecl_server::kernel::{CompiledKernel, CubeKernel};
use cubecl_server::logging::ServerLogger;

use crate::CpuCompiler;
use crate::compute::cpu_kernel::CpuCompiledKernel;

/// The alignment, in bytes, every buffer of a launch is known to start on.
///
/// Storage bases and pool offsets are 64-byte aligned, but a view can weaken
/// that, so a kernel is compiled once per alignment its launches guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferAlignment(pub u32);

/// Compiles kernels for the CPU and loads them.
///
/// Both halves of the compilation are this one type: there is no compilation
/// store, and loading a JIT engine is handing it over.
#[derive(Debug)]
pub(crate) struct CpuKernelCompiler {
    pub options: PlironOptions,
}

impl ArtifactCompiler for CpuKernelCompiler {
    type Variant = BufferAlignment;
    type Lowered = CompiledKernel<PlironCompiler>;
    type Artifact = Arc<CompiledKernel<PlironCompiler>>;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<BufferAlignment>,
        recording: &mut CompilationRecording,
        _logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        let options = PlironOptions {
            cpu_buffer_alignment: Some(id.variant.0),
            ..self.options.clone()
        };
        let compiled = CompiledKernel::lower(kernel, recording, |definition| {
            CompiledKernel::compile(kernel, definition, &mut CpuCompiler::default(), &options)
        })?;
        // The executable artifact here is the JIT engine the compiler built,
        // not the text. A precompiled kernel brings text and no engine.
        if compiled.repr.is_none() {
            return Err(CompilationError::Generic {
                reason: format!(
                    "the CPU runtime cannot load the precompiled kernel `{}`: it runs compiled IR, not source text",
                    kernel.name()
                ),
                backtrace: BackTrace::capture(),
            }
            .into());
        }
        Ok(compiled)
    }

    fn finalize(
        &self,
        _id: &ArtifactId<Self::Variant>,
        lowered: Self::Lowered,
    ) -> Result<Self::Artifact, LaunchError> {
        Ok(Arc::new(lowered))
    }
}

impl CompilationTarget for CpuKernelCompiler {
    type Compiler = Self;
    type Loaded = CpuCompiledKernel;

    fn compiler(&self) -> &Self {
        self
    }

    fn persists(&self) -> bool {
        false
    }

    fn stored(
        &mut self,
        _id: &ArtifactId<BufferAlignment>,
    ) -> Option<<Self as ArtifactCompiler>::Artifact> {
        None
    }

    fn load(
        &mut self,
        _id: &ArtifactId<BufferAlignment>,
        artifact: &<Self as ArtifactCompiler>::Artifact,
    ) -> Result<CpuCompiledKernel, CompilationError> {
        Ok(CpuCompiledKernel::new(artifact.clone()))
    }

    fn store(
        &mut self,
        _id: &ArtifactId<BufferAlignment>,
        _artifact: <Self as ArtifactCompiler>::Artifact,
        _source: Option<&str>,
    ) -> bool {
        false
    }
}

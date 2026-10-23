//! How the CPU runtime compiles a kernel: through pliron and LLVM to a JIT
//! engine, specialized for the alignment of the buffers it is launched on.

use std::sync::Arc;

use cubecl_core::server::LaunchError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_llvm::{PlironCompiler, PlironOptions};
#[cfg(compilation_cache)]
use cubecl_server::compiler::ArtifactStore;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording, CompilationTarget,
};
use cubecl_server::kernel::{CompiledKernel, CubeKernel};
use cubecl_server::logging::ServerLogger;

use crate::CpuCompiler;
#[cfg(compilation_cache)]
use crate::compute::cache::CpuCacheEntry;
use crate::compute::cpu_kernel::CpuCompiledKernel;

/// The alignment, in bytes, every buffer of a launch is known to start on.
///
/// Storage bases and pool offsets are 64-byte aligned, but a view can weaken
/// that, so a kernel is compiled once per alignment its launches guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferAlignment(pub u32);

/// The compiler half that lowers kernels and may run on worker threads.
#[derive(Debug, Clone)]
pub struct CpuArtifactCompiler {
    options: PlironOptions,
}

impl CpuArtifactCompiler {
    pub fn new(options: PlironOptions) -> Self {
        Self { options }
    }
}

impl ArtifactCompiler for CpuArtifactCompiler {
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

/// Compiles kernels for the CPU and loads them.
#[derive(Debug)]
pub struct CpuKernelCompiler {
    compiler: CpuArtifactCompiler,
    #[cfg(compilation_cache)]
    store: ArtifactStore<CpuCacheEntry>,
}

impl CpuKernelCompiler {
    pub fn new(
        options: PlironOptions,
        #[cfg(compilation_cache)] store: ArtifactStore<CpuCacheEntry>,
    ) -> Self {
        #[cfg(compilation_cache)]
        let mut options = options;
        #[cfg(compilation_cache)]
        {
            options.keep_object = store.is_enabled();
        }
        Self {
            compiler: CpuArtifactCompiler::new(options),
            #[cfg(compilation_cache)]
            store,
        }
    }
}

impl CompilationTarget for CpuKernelCompiler {
    type Compiler = CpuArtifactCompiler;
    type Loaded = CpuCompiledKernel;

    fn compiler(&self) -> &Self::Compiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        #[cfg(compilation_cache)]
        {
            self.store.persists()
        }
        #[cfg(not(compilation_cache))]
        {
            false
        }
    }

    fn stored(
        &mut self,
        id: &ArtifactId<BufferAlignment>,
    ) -> Option<<Self::Compiler as ArtifactCompiler>::Artifact> {
        #[cfg(compilation_cache)]
        {
            let entry = self.store.take_variant(id)?;

            let requirements = cubecl_llvm::KernelRequirements {
                needs_parallelism: entry.needs_parallelism,
                shared_memories: entry.shared_memories.clone().into(),
            };

            match cubecl_llvm::PlironEngine::from_object(
                &entry.object,
                &entry.entrypoint_name,
                requirements,
                entry.io.clone().unwrap_or_default(),
            ) {
                Ok(engine) => {
                    log::trace!("Using the compilation store for CPU kernel {:?}", id.kernel);
                    let compiled = CompiledKernel {
                        entrypoint_name: entry.entrypoint_name.clone(),
                        debug_name: None,
                        source: String::new(),
                        repr: Some(cubecl_llvm::PlironArtifact::Jit(engine)),
                        cube_dim: entry.cube_dim,
                        io: entry.io.clone(),
                        debug_info: None,
                    };
                    Some(Arc::new(compiled))
                }
                Err(err) => {
                    log::warn!(
                        "Corrupted or invalid CPU cache entry for kernel {:?}: {err}. Purging entry and recompiling.",
                        id.kernel
                    );
                    self.store.purge_variant(id);
                    None
                }
            }
        }
        #[cfg(not(compilation_cache))]
        {
            let _ = id;
            None
        }
    }

    fn load(
        &mut self,
        _id: &ArtifactId<BufferAlignment>,
        artifact: &<Self::Compiler as ArtifactCompiler>::Artifact,
    ) -> Result<CpuCompiledKernel, CompilationError> {
        Ok(CpuCompiledKernel::new(artifact.clone()))
    }

    fn store(
        &mut self,
        id: &ArtifactId<BufferAlignment>,
        artifact: <Self::Compiler as ArtifactCompiler>::Artifact,
        _source: Option<&str>,
    ) -> bool {
        #[cfg(compilation_cache)]
        {
            if !self.store.is_enabled() {
                return false;
            }

            let Some(cubecl_llvm::PlironArtifact::Jit(engine)) = artifact.repr.as_ref() else {
                return false;
            };

            let Some(object) = engine.take_object_bytes() else {
                return false;
            };

            let entry = CpuCacheEntry {
                object,
                entrypoint_name: artifact.entrypoint_name.clone(),
                cube_dim: artifact.cube_dim,
                io: artifact.io.clone(),
                shared_memories: (&engine.requirements().shared_memories).into(),
                needs_parallelism: engine.requirements().needs_parallelism,
            };

            self.store.store_variant(id, entry)
        }
        #[cfg(not(compilation_cache))]
        {
            let _ = (id, artifact);
            false
        }
    }
}

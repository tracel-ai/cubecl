//! The half of `OpenCL`'s compilation that needs the server: the source store, and the kernels it
//! hands out.

use crate::compute::artifact::{
    OpenClArtifact, OpenClArtifactCompiler, OpenClCacheEntry, OpenClKernel,
};
use cubecl_server::compiler::{
    ArtifactId, ArtifactRoot, ArtifactStore, CompilationError, CompilationTarget,
};

/// Builds `OpenCL` kernels for one device, from the source store when it holds the kernel and from
/// [`OpenClArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct OpenClPrograms {
    compiler: OpenClArtifactCompiler,
    /// Programs kept between runs, as their source and the device binary the driver built: a
    /// stored kernel is loaded without compiling it again, to `OpenCL` C or by the driver.
    store: ArtifactStore<OpenClCacheEntry>,
}

impl OpenClPrograms {
    /// `namespace` names what the stored sources were compiled for: the platform, device, driver
    /// and compilation options.
    pub(crate) fn new(compiler: OpenClArtifactCompiler, namespace: String) -> Self {
        Self {
            compiler,
            store: ArtifactStore::new(ArtifactRoot::OpenCl, namespace),
        }
    }
}

impl CompilationTarget for OpenClPrograms {
    type Compiler = OpenClArtifactCompiler;
    type Loaded = OpenClKernel;

    fn compiler(&self) -> &OpenClArtifactCompiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        self.store.persists()
    }

    fn stored(&mut self, id: &ArtifactId<()>) -> Option<OpenClArtifact> {
        let entry = self.store.take(&id.kernel)?;
        Some(OpenClArtifact {
            entry,
            kernel: None,
        })
    }

    /// The kernel finalizing built, or, for a source read from the store, the one the driver
    /// builds now.
    fn load(
        &mut self,
        _id: &ArtifactId<()>,
        artifact: &OpenClArtifact,
    ) -> Result<OpenClKernel, CompilationError> {
        match &artifact.kernel {
            Some(kernel) => Ok(kernel.clone()),
            None => self.compiler.load(&artifact.entry),
        }
    }

    fn store(
        &mut self,
        id: &ArtifactId<()>,
        artifact: OpenClArtifact,
        _source: Option<&str>,
    ) -> bool {
        self.store.keep(&id.kernel, artifact.entry, None)
    }
}

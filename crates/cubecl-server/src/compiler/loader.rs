//! The kernels a device has loaded, and the one path that loads more.

use alloc::string::String;

use super::{
    ArtifactCompiler, ArtifactId, CompilationCache, CompilationRecording, CompilationTarget,
    VariantOf,
};
use crate::kernel::CubeKernel;
use crate::logging::ServerLogger;
use crate::server::LaunchError;

/// Every kernel a device has loaded, in front of the backend that compiles and
/// loads the ones it has not.
///
/// A kernel it does not hold is read from the compilation store, or failing
/// that compiled and stored, then loaded — in that order on every backend, and
/// recorded the same way: what differs between backends is the
/// [`CompilationTarget`] it is generic over.
pub struct KernelLoader<T: CompilationTarget> {
    target: T,
    loaded: CompilationCache<ArtifactId<VariantOf<T>>, T::Loaded>,
}

impl<T: CompilationTarget> KernelLoader<T> {
    /// A loader holding nothing yet, bound to the active environment exactly
    /// when `target` [persists](CompilationTarget::persists) what it compiles.
    pub fn new(target: T) -> Self {
        let loaded = match target.persists() {
            true => CompilationCache::bound(),
            false => CompilationCache::unbound(),
        };
        Self { target, loaded }
    }

    /// The backend this loader compiles through.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// The kernel loaded for `id`, if it is.
    pub fn get(&mut self, id: &ArtifactId<VariantOf<T>>) -> Option<&T::Loaded> {
        self.loaded.get(id)
    }

    /// The kernel loaded for `kernel` under `variant`, loading it first if it
    /// is not.
    ///
    /// # Errors
    ///
    /// The validation, compilation or load of `kernel` that failed. Nothing
    /// is kept, so the next launch of the kernel tries again.
    pub fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        variant: VariantOf<T>,
        logger: &ServerLogger,
    ) -> Result<&T::Loaded, LaunchError> {
        let id = ArtifactId {
            kernel: kernel.id(),
            variant,
        };
        if !self.loaded.contains(&id) {
            let loaded = self.compile(kernel, &id, logger)?;
            self.loaded.insert(id.clone(), loaded);
        }
        Ok(self.loaded.get(&id).expect("loaded right above"))
    }

    /// Obtains `id`'s artifact from the store or the compiler, and loads it.
    fn compile(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<VariantOf<T>>,
        logger: &ServerLogger,
    ) -> Result<T::Loaded, LaunchError> {
        let mut recording = CompilationRecording::new(&id.kernel);

        if let Some(artifact) = self.target.stored(id) {
            let loaded = self.target.load(id, &artifact)?;
            recording.loaded();
            return Ok(loaded);
        }

        let lowered = self
            .target
            .compiler()
            .lower(kernel, id, &mut recording, logger)?;
        let source = T::Compiler::source(&lowered).map(String::from);

        if let Some(source) = source.as_deref()
            && let Some(artifact) = self.target.stored_for_source(source)
        {
            let loaded = self.target.load(id, &artifact)?;
            let stored = self.target.store(id, artifact, Some(source));
            recording.rekeyed(stored);
            return Ok(loaded);
        }

        let artifact = self.target.compiler().finalize(id, lowered)?;
        let loaded = self.target.load(id, &artifact)?;
        let stored = self.target.store(id, artifact, source.as_deref());
        recording.compiled(stored);
        Ok(loaded)
    }
}

impl<T: CompilationTarget + core::fmt::Debug> core::fmt::Debug for KernelLoader<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The loaded kernels are driver handles, which say nothing in a log.
        f.debug_struct("KernelLoader")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

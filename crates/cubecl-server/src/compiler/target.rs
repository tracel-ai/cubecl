//! What a backend contributes to compiling a kernel, split by where each half
//! may run: [`ArtifactCompiler`] on any thread, so several kernels compile at
//! once; [`CompilationTarget`] only where the server does, since it holds the
//! compilation store and loads on the server's context.

use core::fmt::Debug;
use core::hash::Hash;

use super::{CompilationError, CompilationRecording};
use crate::id::KernelId;
use crate::kernel::CubeKernel;
use crate::logging::ServerLogger;
use crate::server::LaunchError;

/// Which artifact a launch needs: its kernel, and what the launch adds to it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactId<V> {
    /// The kernel.
    pub kernel: KernelId,
    /// What the launch decides about the artifact that the kernel id does not:
    /// the buffer alignment the code is specialized for, the binding layout a
    /// pipeline is built with. `()` when the kernel id says everything.
    pub variant: V,
}

/// The half of a backend's compilation that may run on any thread: from a
/// kernel to the artifact the device loads, in two steps — a lowering pass
/// through cubecl's compiler, then whatever finishes the artifact — with one
/// chance between them to reuse an artifact already finalized. A backend
/// whose device lets several threads create objects at once may use it here.
///
/// `Sync` because it is shared by every thread compiling at once.
pub trait ArtifactCompiler: Sync {
    /// See [`ArtifactId::variant`].
    type Variant: Clone + Eq + Hash + Debug + Send + Sync;
    /// A kernel lowered by cubecl's compiler for this backend, not yet
    /// finalized into what the device loads.
    type Lowered: Send;
    /// What the device loads. The compilation store keeps it, or the part of
    /// it that outlives the process: a pipeline the driver built is rebuilt
    /// from the kept source.
    type Artifact: Send;

    /// Validates `kernel` against the device, expands it to IR and lowers it
    /// with cubecl's compiler for this backend, telling `recording` its IR and
    /// source and `logger` the compilation.
    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<Self::Variant>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError>;

    /// The text an artifact finalized from `lowered` can be reused by: the
    /// source finalizing compiles, when that is slow enough that an artifact
    /// another kernel id already finalized from the same text is worth
    /// reusing — a C++ source two kernel ids expand to, or one expanded again
    /// by a rebuild that changed the build id.
    ///
    /// `None`, the default, finalizes every time. It says nothing about
    /// whether `lowered` has text: a backend whose lowering already produced
    /// the artifact, LLVM's, has text and nothing to save by reusing it.
    fn source(lowered: &Self::Lowered) -> Option<&str> {
        let _ = lowered;
        None
    }

    /// Turns `lowered`, lowered for `id`, into what the device loads:
    /// compiles its source with the platform's compiler, or takes out the
    /// artifact lowering produced.
    ///
    /// A backend whose driver compiles again once handed the artifact, and
    /// lets several threads do so, does that here too and carries the result
    /// in the artifact, so the driver's compilations run alongside each other
    /// rather than one by one in [`CompilationTarget::load`].
    fn finalize(
        &self,
        id: &ArtifactId<Self::Variant>,
        lowered: Self::Lowered,
    ) -> Result<Self::Artifact, LaunchError>;
}

/// The variant an [`ArtifactCompiler`] keys its artifacts by.
pub type VariantOf<T> = <<T as CompilationTarget>::Compiler as ArtifactCompiler>::Variant;
/// The artifact an [`ArtifactCompiler`] produces.
pub type ArtifactOf<T> = <<T as CompilationTarget>::Compiler as ArtifactCompiler>::Artifact;

/// The half of a backend's compilation that runs only where the server does:
/// the compilation store, and the loads on the server's context.
pub trait CompilationTarget {
    /// The half that may run on any thread.
    type Compiler: ArtifactCompiler;
    /// What a launch dispatches. Cloned on every launch, so it should be
    /// handles and shared pointers.
    type Loaded: Clone;

    /// The half that may run on any thread.
    fn compiler(&self) -> &Self::Compiler;

    /// Whether artifacts are kept between runs, which is what binds the
    /// loaded kernels to the active environment: see [`CompilationCache`].
    ///
    /// [`CompilationCache`]: super::CompilationCache
    fn persists(&self) -> bool;

    /// Takes the artifact the compilation store holds for `id` out of it, to
    /// be loaded; `None` when there is no store or it holds nothing for `id`.
    fn stored(&mut self, id: &ArtifactId<VariantOf<Self>>) -> Option<ArtifactOf<Self>>;

    /// Takes the artifact the compilation store holds for another kernel
    /// finalized from `source` out of it — see
    /// [`ArtifactCompiler::source`] — for [`store`](Self::store) to
    /// put back under the kernel asking. `None`, the default, for a backend
    /// that keeps no artifact by source.
    fn stored_for_source(&mut self, source: &str) -> Option<ArtifactOf<Self>> {
        let _ = source;
        None
    }

    /// Loads `artifact` on the device.
    fn load(
        &mut self,
        id: &ArtifactId<VariantOf<Self>>,
        artifact: &ArtifactOf<Self>,
    ) -> Result<Self::Loaded, CompilationError>;

    /// Keeps a loaded `artifact` in the compilation store under `id` and, when
    /// given, under the `source` it was finalized from. Whether the store took
    /// it, as [`store_compiled`](super::store_compiled) answers: `false` with
    /// no store.
    ///
    /// Called after [`load`](Self::load), so an artifact the device refused is
    /// never handed back on the next run.
    fn store(
        &mut self,
        id: &ArtifactId<VariantOf<Self>>,
        artifact: ArtifactOf<Self>,
        source: Option<&str>,
    ) -> bool;
}

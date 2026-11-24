use cubecl_environment::environment::{EnvironmentError, PruneOptions, Pruned};

/// Pruning an environment: removing what this build's work never launched.
///
/// A trait rather than a method, because an
/// [`EnvironmentId`](cubecl_environment::environment::EnvironmentId) is declared
/// where the layout of the compilation stores is unknown: this crate reads it.
pub trait Prune {
    /// Remove the kernels this build compiled and its work never launched,
    /// those another build compiled, and the build records `options` does not
    /// keep. The space they took is reclaimed by the next save, which copies
    /// what is left.
    ///
    /// Prune from the build that runs the environment, once its work has run
    /// with its tunes settled: a work launch made while a key is still
    /// untuned runs whichever candidate comes first, and a pruned environment
    /// keeps that candidate rather than the pick a later tune measures.
    ///
    /// # Errors
    ///
    /// The environment lives in another process's storage, or its database
    /// does not open.
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError>;
}

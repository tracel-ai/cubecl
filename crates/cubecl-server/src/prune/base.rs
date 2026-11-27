use cubecl_environment::environment::{EnvironmentError, PruneOptions, Pruned};

/// Pruning an environment: removing what this build's work never launched.
///
/// A trait rather than a method, because an
/// [`EnvironmentId`](cubecl_environment::environment::EnvironmentId) is declared
/// where the layout of the compilation stores is unknown: this crate reads it.
///
/// # Assumptions
///
/// - It runs from the build that ran the environment's work: the kernels it
///   keeps are the ones this build recorded its work launching.
/// - That work ran with its tunes settled: a work launch made while a key is
///   still untuned runs whichever candidate comes first, and a pruned
///   environment keeps that candidate rather than the pick a later tune
///   measures.
///
/// # Guarantees
///
/// - Every variant of a kernel this build compiled and recorded its work
///   launching stays.
/// - When this build recorded no launch, every kernel it compiled stays.
/// - The autotune answers, and every namespace that is neither a compiler's,
///   a record of what a build's work used nor a build record, stay.
pub trait Prune {
    /// Remove the kernels this build compiled and its work never launched,
    /// those another build compiled with other builds' records of their
    /// work, and the build records `options` does not keep. The space they
    /// took is reclaimed by the next save, which copies what is left.
    ///
    /// # Errors
    ///
    /// The environment lives in another process's storage, or its database
    /// does not open.
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError>;
}

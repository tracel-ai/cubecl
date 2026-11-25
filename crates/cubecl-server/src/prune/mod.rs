//! Removing from an environment what this build's work never used.
//!
//! A tune races every candidate it measures, at the shape it tunes for, and
//! each candidate compiles kernels for that shape; the work then launches the
//! pick's kernels, for every shape it runs. What decides a kernel's fate is
//! therefore not the tune but the work: the
//! [loader](crate::compiler::KernelLoader) records each kernel the work
//! launches ([`WorkLaunchStore`](crate::compiler::WorkLaunchStore)), whatever shape or variant it was launched
//! for, and never one only a race launched. A prune keeps what this build
//! recorded and removes the rest of this build's compiled kernels, every
//! kernel another build compiled with the records of what its work launched,
//! and the build records the caller does not keep.
//!
//! The autotune answers go the same way. An environment may hold more of them
//! than its work needs, brought from elsewhere or left by another version, and
//! each tuner records those it [loads or
//! measures](cubecl_environment::environment::UsedAnswers): a prune keeps what
//! this build recorded, and removes every other answer with other builds'
//! records. A key whose kernels were only gathered has no answer, and an
//! answer a gathering pass loads is recorded as one the work loads.
//!
//! "Build" is the compiled program, as its [`BuildStamp`](cubecl_environment::environment::BuildStamp) names it: a kernel is
//! stored under the build that compiled it, and another build's is loaded
//! only where a backend reuses an artifact by its source, as CUDA's and HIP's
//! do; such a kernel compiles again after a prune. An environment in which
//! this build's work launched nothing keeps every kernel this build compiled,
//! and one in which it used no answer keeps every answer.
//!
//! [`Prune`] is declared on every target, so a caller names it without a
//! condition of its own; an [`EnvironmentId`] implements it where compiled
//! kernels are kept between runs: natively, with `persistence`.
//!
//! [`EnvironmentId`]: cubecl_environment::environment::EnvironmentId

#[cfg(compilation_cache)]
mod answers;
mod base;
#[cfg(compilation_cache)]
mod environment;
#[cfg(compilation_cache)]
mod launches;

pub use base::*;

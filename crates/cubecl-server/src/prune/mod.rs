//! Removing from an environment what this build's work never launched.
//!
//! A tune races every candidate it measures, at the shape it tunes for, and
//! each candidate compiles kernels for that shape; the work then launches the
//! pick's kernels, for every shape it runs. What decides a kernel's fate is
//! therefore not the tune but the work: the
//! [loader](crate::compiler::KernelLoader) records each kernel the work
//! launches ([`WorkLaunchStore`]), whatever shape or variant it was launched
//! for, and never one only a race launched. A prune keeps what this build
//! recorded and removes the rest of this build's compiled kernels, every
//! kernel another build compiled with the records of what its work launched,
//! and the build records the caller does not keep. The autotune answers stay.
//!
//! "Build" is the compiled program, as [`build_id_hash`] names it: a kernel is
//! stored under the build that compiled it, and another build's is loaded
//! only where a backend reuses an artifact by its source, as CUDA's and HIP's
//! do; such a kernel compiles again after a prune. An environment in which
//! this build's work launched nothing keeps every kernel this build compiled.

mod base;
mod launches;

pub use base::*;

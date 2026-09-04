//! `CubeCL`: what a kernel author compiles against.
//!
//! This crate carries no runtime. It is the language, the launch API and the
//! client, and stays free of every runtime crate so that an edit to one
//! rebuilds no kernel. A binary picks the runtimes it links through
//! `cubecl-dispatch`, which is also where a [`Device`] turns into a
//! [`Client`](client::Client).

pub use cubecl_core::*;

pub use cubecl_ir::features;
pub use cubecl_runtime::config;
pub use cubecl_runtime::memory_management::MemoryAllocationMode;

/// Ship pre-warmed autotune and compilation caches with an application.
///
/// Run the application once so its caches are warm, then save the active
/// environment to a file and ship it. On the target machine either
/// [`environment::load`] mounts the file in place, or [`import`] copies it
/// into the local environment, after which the file can be deleted.
///
/// The exporting binary must be built against the same cubecl version as the
/// consuming one. The version is part of every cache namespace, so a bundle
/// built elsewhere installs cleanly and is then ignored, with a warning as the
/// only signal. Calling this from your own crate is what keeps the two in
/// step.
///
/// ```no_run
/// use cubecl::bundle::BundleFormat;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // ... run the work you want tuned and compiled, then:
/// let manifest = cubecl::environment::bundle().save("h100.bundle", BundleFormat::Sqlite)?;
/// println!("exported {}", manifest.name);
/// # Ok(())
/// # }
/// ```
///
/// Merging several roots or restricting the namespaces goes through
/// [`export`] directly.
///
/// [`export`]: cubecl_environment::bundle::export
/// [`import`]: cubecl_environment::bundle::import
/// [`environment::load`]: cubecl_environment::environment::load
pub use cubecl_environment::bundle;

/// Which named environment caches are warmed into, and where it lives.
pub use cubecl_environment::environment;

/// Running a workload for the compilation and tuning it provokes, without
/// running the workload itself.
///
/// This is what makes producing a [`bundle`] affordable: inside a
/// [`DryRun`](dry_run::DryRun) every launch is compiled, cached and tuned
/// without also being executed. Buffers are left as they were, so it only suits
/// a pass driven by the *shapes* it produces.
///
/// ```no_run
/// # fn warm_up() {}
/// let _dry_run = cubecl::dry_run::DryRun::new();
/// warm_up();
/// ```
pub use cubecl_runtime::dry_run;

/// Watching what the runtime runs: the launch observer, and the profiling
/// logger's levels.
///
/// Re-exported because a caller attributing kernels to its own work reaches for
/// [`LaunchObservation`](cubecl_runtime::logging::LaunchObservation) and has no
/// other reason to depend on `cubecl-runtime` directly.
pub use cubecl_runtime::logging;

/// A device of any runtime this build enables, and the runtime it belongs to.
///
/// A value names both the runtime and the device on it, without naming the
/// runtime crate. Turning it into a [`Client`](client::Client) is the runtime's
/// job, and lives in `cubecl-dispatch`.
pub use cubecl_runtime::device::{Device, RuntimeId};

#[cfg(feature = "stdlib")]
pub use cubecl_std as std;

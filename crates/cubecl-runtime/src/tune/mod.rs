//! # Autotuning
//!
//! Autotuning runs several candidate kernels on reference inputs and caches the fastest
//! one per key.
//!
//! ```ignore
//! #[derive(AutotuneKey)]
//! struct KernelKey { size: u32 }
//!
//! fn run_kernel_tuned(lhs: Tensor, rhs: Tensor) -> Tensor {
//!     static TUNER: LocalTuner<String, KernelKey> = local_tuner!();
//!
//!     let tunables = TUNER.init(|| {
//!         TunableSet::new(KernelKey::new, |_key, (lhs, rhs)| (lhs.clone(), rhs.clone()))
//!             .with(Tunable::new("k1", |(lhs, rhs)| kernel_1(lhs, rhs)))
//!             .with(Tunable::new("k2", |(lhs, rhs)| kernel_2(lhs, rhs)))
//!     });
//!
//!     TUNER.execute(&device_id, &lhs.client, tunables, (lhs, rhs));
//! }
//! ```
//!
//! Kernels are closures returning `Result<Out, impl Into<String>>`. Multi-input kernels
//! take a single tuple argument and destructure: `|(lhs, rhs, out)| body`.
//!
//! See [`TuneInputs`] for the borrowed-inputs story, and [`Tunable::new`] for why its
//! HRTB bound is spelled out directly (closure inference).

mod base;
mod bounds_generator;
mod device_access;
mod eviction;
mod gathering;
mod input_generator;
mod key_generator;
mod local;
mod log;
mod operation;
// What a tune leaves in the environment beside its answer.
#[cfg(persistence)]
mod record;
// The adaptive strategy, which only the native driver can run.
#[cfg(not(target_family = "wasm"))]
mod patience;
#[cfg(not(target_family = "wasm"))]
mod sampler;
#[cfg(not(target_family = "wasm"))]
mod schedule;
mod stage;
mod tune_benchmark;
mod tune_cache;
mod tune_inputs;
mod tuner;
mod util;

pub use crate::local_tuner;
pub use crate::throughput::ResourceBound;
pub use base::{OrderedSettings, Patience, Placement, Tunable, TuneGroup};
pub use bounds_generator::{AutotuneBound, Bounds, BoundsGenerator, Thresholds, TimeBound};
// `cubecl-common` cannot depend on `cubecl-runtime`, so `Work` lives there and
// autotune bounds and benchmark reporting share one definition.
pub use cubecl_common::work::Work;
pub use device_access::TuneDeviceAccess;
pub use eviction::{Eviction, Evictor};
pub use input_generator::{CloneInputGenerator, InputGenerator};
pub use key_generator::KeyGenerator;
pub use local::LocalTuner;
#[cfg(std_io)]
pub use log::AutotuneDecision;
pub use log::{AutotuneLogContext, AutotuneLogEvent, AutotuneLoggerExt, CheckResult};
pub use operation::{AutotuneKey, TunableSet, TuneFn};
#[cfg(persistence)]
pub use record::{PlannedCandidate, Trial, TuneRecord};
pub use stage::{StageEnterer, TuneStage};
pub use tune_benchmark::AutotuneOutput;
pub use tune_cache::{AutotuneResult, LAYOUT_PREFIX, TuneCacheResult};
#[cfg(persistence)]
pub use tune_cache::{PersistentCacheKey, PersistentCacheValue, TuneTableName};
pub use tune_inputs::TuneInputs;
pub use tuner::{AutotuneError, AutotuneOutcome, Tuner};
pub use util::anchor;

pub(crate) use base::TunePlan;
pub(crate) use tune_benchmark::tune_benchmark;
pub(crate) use tune_cache::TuneCache;
#[cfg(feature = "autotune-checks")]
pub(crate) use tuner::check_autotune_outputs;

//! # Autotuning
//!
//! Autotuning runs several candidate kernels on reference inputs and caches the fastest
//! one per key.
//!
//! ```ignore
//! #[derive(AutotuneKeyable)]
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

mod anchor;
mod binding;
mod bounds;
mod bounds_generator;
mod candidate_planner;
mod candidate_sampler;
mod checkable;
#[cfg(feature = "autotune-checks")]
mod checks;
mod clone_input_generator;
mod committer;
mod device_access;
mod driver;
mod error;
mod evictor;
mod gathering;
mod group;
mod input_generator;
mod key_generator;
mod keyable;
mod local;
mod log;
mod membership;
mod pending_unmarker;
#[cfg(persistence)]
mod persister;
mod plan;
// What a tune leaves in the environment beside its answer.
#[cfg(persistence)]
mod record;
mod sample_time;
mod set_initializer;
mod settler;
mod stage;
mod tunable;
mod tunable_set;
mod tune_benchmark;
mod tune_cache;
mod tune_executor;
mod tune_fn;
mod tune_inputs;
mod tune_logger;
mod tune_stage;
mod tuner;
mod tuner_opener;
// The adaptive strategy, which only the native driver can run.
#[cfg(not(target_family = "wasm"))]
mod first_pass;
#[cfg(not(target_family = "wasm"))]
mod patience;
#[cfg(not(target_family = "wasm"))]
mod sampler;
#[cfg(not(target_family = "wasm"))]
mod schedule;

pub use crate::local_tuner;
pub use crate::throughput::ResourceBound;
pub use anchor::Anchor;
pub use bounds::{AutotuneBound, Bounds, Thresholds};
pub use bounds_generator::BoundsGenerator;
pub use checkable::AutotuneCheckable;
// `cubecl-common` cannot depend on `cubecl-runtime`, so `Work` lives there and
// autotune bounds and benchmark reporting share one definition.
pub use cubecl_common::work::Work;
pub use device_access::TuneDeviceAccess;
pub use error::{AutotuneError, AutotuneOutcome};
pub use evictor::Evictor;
pub use group::{OrderedSettings, Patience, TuneGroup};
pub use input_generator::InputGenerator;
pub use key_generator::KeyGenerator;
pub use keyable::AutotuneKeyable;
pub use local::LocalTuner;
#[cfg(std_io)]
pub use log::AutotuneDecision;
pub use log::{AutotuneLogContext, AutotuneLogEvent, CheckResult};
pub use plan::Placement;
#[cfg(persistence)]
pub use record::{PlannedCandidate, Trial, TuneRecord};
pub use stage::StageEnterer;
pub use tunable::Tunable;
pub use tunable_set::TunableSet;
pub use tune_cache::{AutotuneResult, LAYOUT_PREFIX};
#[cfg(persistence)]
pub use tune_cache::{PersistentCacheKey, PersistentCacheValue, TuneTableName};
pub use tune_inputs::TuneInputs;
pub use tune_stage::TuneStage;

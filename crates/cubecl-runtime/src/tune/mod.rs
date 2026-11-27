//! # Autotuning
//!
//! Autotuning runs several candidate kernels on reference inputs and caches the one it decides,
//! per key.
//!
//! ```ignore
//! #[derive(AutotuneKeyable)]
//! struct KernelKey { size: u32 }
//!
//! fn run_kernel_tuned(device_id: DeviceId, lhs: Tensor, rhs: Tensor) -> Tensor {
//!     static TUNER: LocalTuner<KernelKey, DeviceId> = local_tuner!();
//!
//!     let tunables = TUNER.init(&device_id, || {
//!         TunableSet::new(KernelKey::new, |_key, (lhs, rhs)| (lhs.clone(), rhs.clone()))
//!             .with(Tunable::new("k1", |(lhs, rhs)| kernel_1(lhs, rhs)))
//!             .with(Tunable::new("k2", |(lhs, rhs)| kernel_2(lhs, rhs)))
//!     });
//!
//!     TUNER.execute(&device_id, &lhs.client, tunables, (lhs, rhs))
//! }
//! ```
//!
//! Kernels are closures returning `Result<Out, impl Into<String>>`. Multi-input kernels
//! take a single tuple argument and destructure: `|(lhs, rhs, out)| body`.
//!
//! See [`TuneInputs`] for the borrowed-inputs story, and [`Tunable::new`] for why its
//! HRTB bound is spelled out directly (closure inference).

mod bounds;
mod device_access;
mod error;

#[cfg(not(target_family = "wasm"))]
mod adaptive;
mod cache;
mod drive;
mod execute;
mod hook;
mod key;
mod log;
mod measure;
mod plan;
mod set;
mod tune_inputs;

pub use crate::local_tuner;
pub use crate::throughput::ResourceBound;
pub use bounds::{AutotuneBound, Bounds, Thresholds};
#[cfg(persistence)]
pub use cache::record::{Placement, PlannedCandidate, Trial, TuneRecord};
pub use cache::table::LAYOUT_PREFIX;
#[cfg(persistence)]
pub use cache::table::{PersistentCacheKey, PersistentCacheValue, TuneTableName};
pub use hook::bounds_generator::BoundsGenerator;
pub use key::anchor::Anchor;
pub use set::checkable::AutotuneCheckable;
// `cubecl-common` cannot depend on `cubecl-runtime`, so `Work` lives there and
// autotune bounds and benchmark reporting share one definition.
pub use cubecl_common::work::Work;
pub use device_access::TuneDeviceAccess;
pub use error::AutotuneError;
pub use execute::local_tuner::LocalTuner;
pub use hook::evictor::Evictor;
pub use hook::input_generator::InputGenerator;
pub use hook::key_generator::KeyGenerator;
pub use hook::stage_enterer::StageEnterer;
pub use hook::tune_stage::TuneStage;
pub use key::keyable::AutotuneKeyable;
#[cfg(std_io)]
pub use log::context::AutotuneDecision;
pub use log::context::{AutotuneLogContext, AutotuneLogEvent, CheckResult};
pub use measure::outcome::{AutotuneOutcome, AutotuneResult};
pub use plan::group::{OrderedSettings, Patience, TuneGroup};
pub use set::tunable::Tunable;
pub use set::tunable_set::TunableSet;
pub use tune_inputs::TuneInputs;

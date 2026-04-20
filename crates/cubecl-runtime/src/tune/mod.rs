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
mod input_generator;
mod key_generator;
mod local;
mod operation;
mod tune_benchmark;
mod tune_cache;
mod tune_inputs;
mod tuner;
mod util;

pub use base::*;
pub use input_generator::*;
pub use key_generator::*;
pub use local::*;
pub use operation::*;
pub use tune_benchmark::*;
pub use tune_cache::*;
pub use tune_inputs::*;
pub use tuner::*;
pub use util::*;

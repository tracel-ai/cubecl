//! # Autotuning
//!
//! Autotuning runs several candidate kernels on reference inputs and caches the fastest
//! one per key. Kernels register as [`TuneFn`] impls (usually via a plain function or
//! closure — the conversion is automatic).
//!
//! # Example
//!
//! ```ignore
//! #[derive(AutotuneKey)]
//! struct KernelKey { size: u32 }
//!
//! fn run_kernel_tuned(lhs: Tensor, rhs: Tensor) -> Tensor {
//!     static TUNER: LocalTuner<String, KernelKey> = local_tuner!();
//!
//!     let tunables = TUNER.init(|| {
//!         TunableSet::new(KernelKey::new, |_key, lhs, rhs| (lhs.clone(), rhs.clone()))
//!             .with(Tunable::new("k1", kernel_1))
//!             .with(Tunable::new("k2", kernel_2))
//!             .with(Tunable::new("k3", kernel_3))
//!     });
//!
//!     TUNER.execute("hello".to_string(), &lhs.client, &tunables, (lhs, rhs));
//! }
//! ```
//!
//! Kernels are closures returning `Result<Out, impl Into<String>>`. Multi-input kernels
//! destructure a tuple: `|(lhs, rhs, out)| body`. Pre-built [`TuneFn`] impls register via
//! [`Tunable::from_impl`].
//!
//! ## Borrowed inputs: the [`TuneInputs`] trait
//!
//! A [`TunableSet`] stores tunables `'static` (so [`LocalTuner::init`] can cache it in an
//! `Arc<dyn Any>`). But callers may want to pass *borrowed* inputs — e.g. burn's fusion
//! tuner threads a `&mut Context<'a, …>` through `TuneInput<'a, R, O>`. To reconcile
//! this, `TunableSet` is parameterized by `I: TuneInputs`, a `'static` marker type whose
//! GAT `I::At<'a>` gives the concrete input type at lifetime `'a`. Every `TuneFn::execute`
//! call is HRTB over `'a`, so a stored `dyn TuneFn<Inputs = I, …>` still accepts
//! `I::At<'a>` for any `'a`.
//!
//! - For `'static` inputs, use [`OwnedInputs<T>`] as a zero-cost marker — `At<'a> = T`,
//!   ignoring the lifetime. Multi-input kernels use a tuple: `OwnedInputs<(A, B, C)>`.
//! - For borrowed inputs, define your own `TuneInputs` impl (see
//!   `burn-cubecl-fusion::tune::FusionTuneInputs`).
//!
//! [`Tunable::new`] spells out the HRTB bound
//! `for<'a> Fn(I::At<'a>) -> Result<Out, Err>` directly in its `where`-clause. This is
//! load-bearing for Rust closure type inference: if the HRTB were hidden behind a
//! helper trait, closure inference would pick a single concrete lifetime for the
//! argument and you'd hit the classic "implementation of FnOnce is not general enough"
//! error on any family whose `At<'a>` actually depends on `'a`.

mod base;
mod tune_inputs;
mod input_generator;
mod key_generator;
mod local;
mod operation;
mod tune_benchmark;
mod tune_cache;
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

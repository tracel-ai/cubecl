//! # Autotuning
//!
//! Autotuning allows running different kernels or comptime parameters to find the fastest one
//! for any given input. Kernels must implement [`TuneFn`](crate::tune::TuneFn) (see below).
//!
//! # Example
//!
//! ```ignore
//! #[derive(AutotuneKey)]
//! struct KernelKey {
//!     size: u32
//! }
//!
//! fn run_kernel_tuned(lhs: Tensor, rhs: Tensor) -> Tensor {
//!     static TUNER: LocalTuner<String, KernelKey> = local_tuner!();
//!     
//!     let tunables = TUNER.init(|| {
//!         TunableSet::new(KernelKey::new, |_key, lhs, rhs| (lhs.clone(), rhs.clone()))
//!             .with(Tunable::new(kernel_1))
//!             .with(Tunable::new(kernel_2.ok()))
//!             .with(Tunable::new(kernel_3)
//!     });
//!    
//!     TUNER.execute("hello".to_string(), &lhs.client, &tunables, (lhs, rhs));
//! }
//! ```
//!
//! # Tunable
//!
//! [`TuneFn`](crate::tune::TuneFn) is implemented automatically for all functions and closures
//! that take a set of cloneable inputs, and return a `Result<Out, impl Into<AutotuneError>>`. If the
//! kernel does not return a [`Result`], use `kernel_fn.ok()` to wrap it in `Ok` and turn it into a
//! tunable.
//!
//! ## Implementation details
//!
//! To implement `TuneFn` for all valid tunable functions, a set of patterns is employed.
//! TuneFn functions don't directly implement `TuneFn`, they implement `IntoTuneFn` instead. The
//! reason for this is that the Rust trait resolver can't detect that traits like `Fn(A, B)`
//! and `Fn(A)` are mutually exclusive. This means trying to implement `TuneFn` for both would
//! cause conflicting implementations. To solve this problem, a `Marker` generic is employed, that
//! stores a dummy type (like `IsFunction`), along with the equivalent function pointer of the
//! signature (which is a type, not a trait), allowing the trait resolver to correctly identify
//! the implementations as distinct. However, since different kinds of `TuneFn` will have different
//! `Marker` generics, the `IntoTuneFn` trait is needed to erase the marker.
//! This way, only [`Tunable::new`](crate::tune::Tunable::new) requires the
//! marker as a generic, which it then erases by calling
//! [`IntoTuneFn::into_tunable`](crate::tune::IntoTuneFn::into_tunable).
//! The same technique is used for [`KeyGenerator`](crate::tune::KeyGenerator) and
//! [`InputGenerator`](crate::tune::InputGenerator).
//!
//! The last set of traits are [`AsFunctionTunable`](crate::tune::AsFunctionTunable) and
//! [`AsFunctionTunableResult`](crate::tune::AsFunctionTunableResult). These traits are directly
//! implemented by all tunable functions and allow us to annotate function-like
//! tunables specifically, to allow things like overriding the name, wrapping the return type in
//! `Ok` ([`AsFunctionTunable::ok`](crate::tune::AsFunctionTunable::ok)), and other things. They also help with error messages. This is
//! done by using [`#[diagnostic::on_unimplemented(...)]`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticon_unimplemented-attribute).

mod base;
mod function_tunable;
mod input_generator;
mod key_generator;
mod local;
mod operation;
mod tune_benchmark;
mod tune_cache;
mod tuner;
mod util;

pub use base::*;
pub use function_tunable::*;
pub use input_generator::*;
pub use key_generator::*;
pub use local::*;
pub use operation::*;
pub use tune_benchmark::AutotuneOutput;
pub use tune_benchmark::*;
pub use tune_cache::*;
pub use tuner::*;
pub use util::*;

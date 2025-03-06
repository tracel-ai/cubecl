//! # Autotuning
//!
//! Autotuning allows running different kernels or comptime parameters to find the fastest one
//! for any given input. Kernels must implement [`Tunable`](crate::tune::Tunable) (see below).
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
//!     let tunables = TunableSet::new(KernelKey::new, |_key, lhs, rhs| (lhs.clone(), rhs.clone()))
//!         .with_tunable(kernel_1)
//!         .with_tunable(kernel_2.ok())
//!         .with_tunable(kernel_3);
//!    
//!     TUNER.execute("hello".to_string(), &lhs.client, &tunables, (lhs, rhs));
//! }
//! ```
//!
//! # Tunable
//!
//! [`Tunable`](crate::tune::Tunable) is implemented automatically for all functions and closures
//! that take a set of cloneable inputs, and return a `Result<Out, impl Into<AutotuneError>>`. If the
//! kernel does not return a [`Result`], use `kernel_fn.ok()` to wrap it in `Ok` and turn it into a
//! tunable.
//!
//! ## Implementation details
//!
//! To implement `Tunable` for all valid tunable functions, a set of patterns is employed.
//! Tunable functions don't directly implement `Tunable`, they implement `IntoTunable` instead. The
//! reason for this is that the Rust trait resolver can't detect that traits like `Fn(A, B)`
//! and `Fn(A)` are mutually exclusive. This means trying to implement `Tunable` for both would
//! cause conflicting implementations. To solve this problem, a `Marker` generic is employed, that
//! stores a dummy type (like `IsFunction`), along with the equivalent function pointer of the
//! signature (which is a type, not a trait), allowing the trait resolver to correctly identify
//! the implementations as distinct. However, since different kinds of `Tunable` will have different
//! `Marker` generics, the `IntoTunable` trait is needed to erase the marker.
//! This way, only [`TunableSet::with_tunable`](crate::tune::TunableSet::with_tunable) requires the
//! marker as a generic, which it then erases by calling
//! [`IntoTunable::into_tunable`](crate::tune::IntoTunable::into_tunable).
//! The same technique is used for [`KeyGenerator`](crate::tune::KeyGenerator) and
//! [`InputGenerator`](crate::tune::InputGenerator).
//!
//! The last set of traits are [`AsFunctionTunable`](crate::tune::AsFunctionTunable) and
//! [`AsFunctionTunableResult`](crate::tune::AsFunctionTunableResult). These traits are directly
//! implemented by all tunable functions and allow us to annotate function-like
//! tunables specifically, to allow things like overriding the name, wrapping the return type in
//! `Ok` ([`AsFunctionTunable::ok`](crate::tune::AsFunctionTunable::ok)), and other things. They also help with error messages. This is
//! done by using [`#[diagnostic::on_unimplemented(...)]`](https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-diagnosticon_unimplemented-attribute).

mod function_tunable;
mod input_generator;
mod key_generator;
mod local;
mod operation;
mod tune_benchmark;
mod tune_cache;
mod tuner;
mod util;

pub use function_tunable::*;
pub use input_generator::*;
pub use key_generator::*;
pub use local::*;
pub use operation::*;
pub use tune_benchmark::*;
pub use tune_cache::*;
pub use tuner::*;
pub use util::*;

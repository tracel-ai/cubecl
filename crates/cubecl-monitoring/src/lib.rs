#![no_std]
#![warn(missing_docs)]

//! How busy a `CubeCL` device is, by the counter its card's driver or its platform keeps.
//!
//! Each GPU counter is a feature, so a runtime pulls in only the ones its cards are read through;
//! with none on, the crate is only the types a runtime reports a reading with.

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

/// A GPU's utilization, by the counter its card's driver or its platform keeps. The card and the
/// platform decide which counter answers, never the runtime: an NVIDIA card reads the same through
/// CUDA as through wgpu.
pub mod gpu_utilization;
mod utilization;

pub use utilization::*;

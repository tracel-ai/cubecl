//! Warpgroup matrix-multiply and accumulate, Hopper's asynchronous tensor core instructions.
//!
//! The four planes of a warpgroup (128 contiguous units, starting at a multiple of 128) issue one
//! `64 x n x k` MMA together. `B`, and optionally `A`, are read straight from shared memory
//! through [matrix descriptors](MatrixDescriptor), and the [`Accumulator`] stays in registers,
//! spread over the warpgroup as [`Accumulator::position_of_nth`] describes.
//!
//! The MMAs run asynchronously, and own the accumulator while they do: [`Accumulator::start`]
//! hands it over as a [`Pending`](crate::prelude::Pending), which issues the MMAs, and gives it back once they complete.
//! A K loop over shared memory stages looks like:
//!
//! ```rust, ignore
//! let layout = WgmmaTileLayout { major: Major::K, swizzle: Swizzle::B128, rows: 64, k: 64 };
//! let mut acc = Accumulator::<f32>::new(128usize).start();
//! for stage in 0..num_stages {
//!     // ... wait for the stage's tiles to land in shared memory ...
//!     let a = MatrixDescriptor::new(&lhs_tile[stage], layout);
//!     let b = MatrixDescriptor::new(&rhs_tile[stage], layout_b);
//!     #[unroll]
//!     for k in 0..steps {
//!         acc.execute(&a.at(0, k * k_step), &b.at(0, k * k_step));
//!     }
//!     // The MMAs of this stage, done once this resolves: wait on it before reusing the stage.
//!     let read = acc.commit();
//!     // ...
//! }
//! let acc = acc.wait();
//! ```
//!
//! Shared memory the units wrote is only visible to the MMAs after
//! [`sync_async_proxy_shared`](fn@crate::prelude::sync_async_proxy_shared) and a cube sync; a TMA
//! load needs neither.
//!
//! Requires [`wgmma`](cubecl_ir::features::MatmulFeatures::wgmma), only present on `sm_90a`.

mod base;
mod layout;

pub use base::*;
pub use layout::{Major, Swizzle, WgmmaLayoutError, WgmmaTileLayout};

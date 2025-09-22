//! Solves full reductions by loading blocks into shared memory.
//! Handles memory movement, bound checks, plane specialization.

pub mod args;
pub mod memory;
pub mod multi_stage;
pub mod read;
pub mod single_stage;

mod base;
mod copy_mechanism;
mod shared;
mod specialization;
mod write;

pub use base::*;
pub use copy_mechanism::*;
pub use shared::*;
pub use specialization::*;
pub use write::*;

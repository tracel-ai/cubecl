//! Solves full reductions by loading blocks into shared memory.
//! Handles memory movement, bound checks, plane specialization.

pub mod args;
pub mod load;
pub mod multi_stage;
pub mod quantization;
pub mod single_stage;
pub mod tensor_view;

mod accumulator_loader;
mod base;
mod copy_mechanism;
mod shared;
mod specialization;
mod write;

pub use accumulator_loader::*;
pub use base::*;
pub use copy_mechanism::*;
pub use quantization::*;
pub use shared::*;
pub use specialization::*;
pub use write::*;

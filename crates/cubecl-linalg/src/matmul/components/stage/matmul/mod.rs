mod stage_matmul_impl;

pub mod plane;
pub(super) mod shared;
pub mod unit;

pub use plane as plane_matmul;
pub use shared::*;
pub use unit as unit_matmul;

pub mod global;
pub mod stage;

mod config;
mod error;
mod problem;
mod selection;

pub use config::*;
use cubecl_matmul::components::tile::{accelerated::AcceleratedMatmul, io::Strided};
use cubecl_std::CubeOption;
pub use error::*;
pub use problem::*;
pub use selection::*;

/// Convolution using `AcceleratedMatmul`
pub type AcceleratedConv = AcceleratedMatmul<CubeOption<Strided>>;

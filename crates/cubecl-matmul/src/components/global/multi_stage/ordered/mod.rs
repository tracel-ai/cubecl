mod config;
mod matmul;
mod setup;

pub use config::*;
pub use matmul::*;
pub use setup::OrderedDoubleBufferingMatmulFamily;

use crate::components::global::read::sync_full_ordered;

/// The ordered double buffering global matmul
/// requires tilewise loading on `Lhs` to guarantee that planes
/// only use data they have loaded themselves.
pub type LL = sync_full_ordered::SyncFullOrderedLoading;

//! One unit of work against the device, and what a backend supplies for it.

mod backend;
mod base;
mod capture;
mod collective;
mod collective_server;
mod peers;
mod staging;
mod stream_relocation_copier;

pub use backend::*;
pub use base::*;
pub use capture::*;
pub use collective::*;
pub use collective_server::*;
pub use peers::*;
pub(crate) use staging::ChunkStager;
pub use staging::{Staging, StagingKind};

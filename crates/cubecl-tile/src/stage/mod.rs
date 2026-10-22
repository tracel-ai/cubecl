//! Staging an operand: storage, transports, pipelined slots and the barriers between fill and read.

pub(crate) mod base;
pub(crate) mod pipeline;
pub(crate) mod rendezvous;
pub(crate) mod staged;
pub(crate) mod transport;

pub(crate) use pipeline::*;
pub(crate) use rendezvous::*;
pub(crate) use staged::StageElement;
pub(crate) use transport::*;

pub use base::{RowChunks, UnitRead};
pub use pipeline::base::Role;
pub use pipeline::held::{AsyncStage, AsyncStageExpand};
pub use pipeline::payload::pair::OperandPair;
pub use pipeline::schedule::Prefetch;
pub use pipeline::slot::Slot;

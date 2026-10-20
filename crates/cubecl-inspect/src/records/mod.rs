//! The records of one environment file, read whole and joined: what the
//! [reports](crate::report) are folded from. Nothing here is a report.

mod compiling;
mod environment;
mod tune_index;

pub(crate) use compiling::CompilationSpans;
pub(crate) use environment::{EnvironmentRecords, SessionRecords};
pub(crate) use tune_index::TuneIndex;

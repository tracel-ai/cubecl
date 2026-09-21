use crate::report::KeyId;
use std::path::PathBuf;

/// Why a report could not be produced.
#[derive(Debug, thiserror::Error)]
pub enum InspectError {
    /// The file is missing, or is not a database cubecl can read.
    #[error("cannot open the environment at {path}: {reason}")]
    Open { path: PathBuf, reason: String },
    /// A directory of environments could not be listed.
    #[error("cannot list {path}: {source}")]
    Directory {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A copy of the environment could not be written.
    #[error("cannot write {path}: {reason}")]
    Export { path: PathBuf, reason: String },
    /// The file's records could not be pruned in place.
    #[error("cannot prune {path}: {reason}")]
    Prune { path: PathBuf, reason: String },
    /// No autotune key in the file carries this id.
    #[error("no autotune key {0} in this environment")]
    UnknownKey(KeyId),
    /// No kernel instance's id starts with this.
    #[error("no kernel {0} in this environment")]
    UnknownKernel(String),
    /// Several kernel instances' ids start with this.
    #[error("`{prefix}` names {count} kernels; give more of the id")]
    AmbiguousKernel { prefix: String, count: usize },
    /// Several builds recorded the one kernel this names.
    #[error("{kernel} was recorded by builds {}; name one with --build", builds.join(", "))]
    AmbiguousBuild { kernel: String, builds: Vec<String> },
    /// A report could not be written out.
    #[error(transparent)]
    Write(#[from] std::io::Error),
    /// A report could not be serialized.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

use serde::Serialize;
use std::time::Duration;

/// What started inside a stretch of one session's clock, a whole session or
/// one of its marks: its tunes and compilations, and what they took.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Activity {
    /// Keys it tuned.
    pub tunes: u64,
    /// The walls of those tunes, summed.
    pub tuning: Duration,
    /// Kernels it compiled or loaded.
    pub compilations: u64,
    /// The time those took: a batch compiled on several threads takes its
    /// wall, not the sum of its kernels' work.
    pub compiling: Duration,
    /// The part of [`compiling`](Self::compiling) that happened inside a
    /// tune, and so is also part of [`tuning`](Self::tuning).
    pub compiling_in_tunes: Duration,
    /// The longest tune.
    pub slowest: Option<SlowTune>,
}

/// The tune a stretch of a session spent longest on.
#[derive(Clone, Debug, Serialize)]
pub struct SlowTune {
    /// The table's namespace.
    pub table: String,
    pub key: ciborium::Value,
    pub wall: Duration,
}

impl Activity {
    /// The compiling that happened outside every tune: what
    /// [`tuning`](Self::tuning) does not already count.
    pub fn compiling_outside_tunes(&self) -> Duration {
        self.compiling.saturating_sub(self.compiling_in_tunes)
    }
}

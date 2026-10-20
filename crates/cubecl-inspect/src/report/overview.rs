use super::{AutotuneReport, CandidateReport, KernelReport, MemorySnapshots, Summary, Timeline};
use serde::Serialize;

/// Every report of a whole environment that takes no argument, read at
/// once: what a reader takes in before selecting anything.
///
/// The reports of one key or one kernel, and those that read beyond the file
/// — a diff, a listing — are not part of it: each needs something an
/// overview does not have.
#[derive(Clone, Debug, Serialize)]
pub struct Overview {
    pub summary: Summary,
    pub timeline: Timeline,
    /// Slowest tune first, as the command line lists them by default.
    pub autotune: AutotuneReport,
    /// Tallied from the same reading as [`autotune`](Self::autotune).
    pub candidates: CandidateReport,
    pub kernels: KernelReport,
    pub memory: MemorySnapshots,
}

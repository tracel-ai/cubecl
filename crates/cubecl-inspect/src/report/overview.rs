use super::{AutotuneReport, CandidateReport, KernelReport, MemorySnapshots, Summary, Timeline};
use serde::Serialize;

/// A report of a whole environment that takes no argument: what one can look
/// at with nothing selected.
///
/// The reports of one key or one kernel, and those that read beyond the file
/// — a diff, a listing — are not overviews: each needs something an overview
/// does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overview {
    /// See [`Summary`].
    Summary,
    /// See [`Timeline`].
    Timeline,
    /// See [`AutotuneReport`], slowest tune first.
    Autotune,
    /// See [`CandidateReport`].
    Candidates,
    /// See [`KernelReport`].
    Kernels,
    /// See [`MemorySnapshots`].
    Memory,
}

/// One [`Overview`], read.
///
/// Untagged, so its JSON is the report's own.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum OverviewReport {
    Summary(Summary),
    Timeline(Timeline),
    Autotune(AutotuneReport),
    Candidates(CandidateReport),
    Kernels(KernelReport),
    Memory(MemorySnapshots),
}

impl Overview {
    /// Every overview, in the order a reader is shown them.
    pub const ALL: [Self; 6] = [
        Self::Summary,
        Self::Timeline,
        Self::Autotune,
        Self::Candidates,
        Self::Kernels,
        Self::Memory,
    ];

    /// The overview's name, as a reader is shown it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::Timeline => "timeline",
            Self::Autotune => "autotune",
            Self::Candidates => "candidates",
            Self::Kernels => "kernels",
            Self::Memory => "memory",
        }
    }
}

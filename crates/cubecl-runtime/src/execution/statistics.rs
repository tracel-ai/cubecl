//! What the overrides opened with a [`StatisticsCollector`](super::StatisticsCollector) triggered: the
//! kernels obtained and the autotune keys tuned, counted to it by the code
//! doing the work.

use crate::id::KernelId;
use crate::server::LaunchError;

/// A snapshot of what a collector's overrides triggered. Every count only
/// grows, over every override and on every device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionStatistics {
    /// The kernels obtained.
    pub compilation: CompilationStatistics,
    /// The autotune keys tuned.
    pub autotune: AutotuneStatistics,
}

/// The kernels the launches under a collector's overrides obtained.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompilationStatistics {
    /// Queued, or asked for by a launch that found them neither loaded nor
    /// queued. A kernel tried again after failing registers again.
    pub registered_count: usize,
    /// Compiled by the backend's compiler.
    pub compiled_count: usize,
    /// Read from the compilation store instead.
    pub loaded_count: usize,
    /// Failed to compile.
    pub failed_count: usize,
    /// Dropped before it settled: the batch compiling it panicked, or the
    /// server holding it went away.
    pub abandoned_count: usize,
    /// Compiled or read from the store, then refused by the device when it
    /// was loaded: counted among [`compiled_count`](Self::compiled_count) or
    /// [`loaded_count`](Self::loaded_count) too, since those are counted as
    /// the batch obtains them, before anything is loaded.
    pub refused_count: usize,
    /// Kept in the compilation store once loaded: what the environment grew
    /// by. Read from the store under its own key, or refused, a kernel is not
    /// stored again; one taken from the store by its source is, under its
    /// own key.
    pub stored_count: usize,
}

impl CompilationStatistics {
    /// The kernels done, however they ended: never more than
    /// [`registered_count`](Self::registered_count).
    pub fn settled_count(&self) -> usize {
        self.compiled_count + self.loaded_count + self.failed_count + self.abandoned_count
    }
}

/// The autotune keys tuned under a collector's overrides. A key with one
/// candidate is answered, not tuned, and counts nowhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutotuneStatistics {
    /// Gathered by a [`CompileOnly`](super::ProcessMode::CompileOnly)
    /// pass, or reached ungathered by a tune that measures or a read of the
    /// persistent autotune cache. A key reached inside another key's
    /// candidates is not gathered: it registers if it is ever measured or read.
    pub registered_count: usize,
    /// Tuned by measuring their candidates.
    pub measured_count: usize,
    /// Read from the persistent autotune cache instead of measured.
    pub loaded_count: usize,
    /// Every candidate failed: the pick was made unmeasured.
    pub failed_count: usize,
    /// Dropped before it settled: a candidate panicked, the device was lost
    /// while it measured, or an environment switch dropped the key it was
    /// gathered for.
    pub abandoned_count: usize,
    /// Measured, and kept in the persistent autotune cache: what the
    /// environment grew by. An unmeasured pick is never kept.
    pub persisted_count: usize,
}

impl AutotuneStatistics {
    /// The keys done, however they ended: never more than
    /// [`registered_count`](Self::registered_count).
    pub fn settled_count(&self) -> usize {
        self.measured_count + self.loaded_count + self.failed_count + self.abandoned_count
    }
}

/// A kernel the work needed that failed to compile or to load under a
/// collector's overrides: one its launches will fail on whenever they run.
///
/// An autotune candidate that fails is dropped from its race and is no
/// failure of the work: it is counted [`failed_count`](CompilationStatistics::failed_count)
/// and listed nowhere — raced, or run
/// [provisionally](super::LaunchPurpose::Provisional) before its tune decided. Only a kernel launched for the
/// [work](super::LaunchPurpose::Work) — a plain launch, the pick of a tune —
/// is listed, so a pass whose launches are discarded still hears that a
/// kernel it reached can never run.
#[derive(Debug, Clone)]
pub struct WorkCompilationFailure {
    /// The kernel that failed.
    pub kernel: KernelId,
    /// Why it failed.
    pub error: LaunchError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_count_sums_every_way_a_kernel_ends() {
        let statistics = CompilationStatistics {
            compiled_count: 1,
            loaded_count: 2,
            failed_count: 3,
            abandoned_count: 4,
            refused_count: 5,
            ..CompilationStatistics::default()
        };
        assert_eq!(statistics.settled_count(), 10);
    }

    #[test]
    fn a_settled_count_sums_every_way_a_key_ends() {
        let statistics = AutotuneStatistics {
            measured_count: 1,
            loaded_count: 2,
            failed_count: 3,
            abandoned_count: 4,
            persisted_count: 5,
            ..AutotuneStatistics::default()
        };
        assert_eq!(statistics.settled_count(), 10);
    }
}

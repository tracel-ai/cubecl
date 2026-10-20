use crate::report::UntypedTuneRecord;
use cubecl_environment::records::Stamped;
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord};
use std::ops::RangeBounds;
use std::time::Duration;

/// Compilations of one session, as the stretches of its clock they took:
/// what tells the time compiling took apart from the work it did.
///
/// A server obtains kernels in batches, compiling a batch's kernels on
/// several threads at once, and records each batch beside its kernels, all
/// stamped when the batch started. A batch's wall is the time it took; its
/// kernels' work adds up every thread's. A session recorded before cubecl
/// recorded batches has none: each of its kernels compiled alone, so a
/// kernel's work is the time it took.
#[derive(Clone, Debug, Default)]
pub struct CompilationSpans {
    /// Each stretch's start on the session clock, and its wall.
    spans: Vec<(Duration, Duration)>,
}

impl CompilationSpans {
    /// The stretches of one session's `batches`, or of its `trips` when it
    /// recorded no batch.
    pub fn new<'a>(
        batches: impl IntoIterator<Item = &'a Stamped<CompilationBatchRecord>>,
        trips: impl IntoIterator<Item = &'a Stamped<CompilationRecord>>,
    ) -> Self {
        let mut spans: Vec<(Duration, Duration)> = batches
            .into_iter()
            .map(|batch| (batch.stamp.offset, batch.record.wall))
            .collect();
        if spans.is_empty() {
            spans = trips
                .into_iter()
                .map(|trip| (trip.stamp.offset, trip.record.work))
                .collect();
        }
        Self { spans }
    }

    /// The time the compilations that started inside `window` took; `..`
    /// for every one.
    pub fn within(&self, window: &impl RangeBounds<Duration>) -> Duration {
        self.spans
            .iter()
            .filter(|(start, _)| window.contains(start))
            .map(|&(_, wall)| wall)
            .sum()
    }

    /// The time the compilations that started during `tune` took: the tuner
    /// waits on the server while it compiles, so a compilation stamped inside
    /// the tune's span on the same session clock is the tune's.
    pub fn during(&self, tune: &Stamped<UntypedTuneRecord>) -> Duration {
        let start = tune.stamp.offset;
        self.within(&(start..start + tune.record.wall))
    }

    /// When the last compilation ended, on the session clock.
    pub fn end(&self) -> Duration {
        self.spans
            .iter()
            .map(|&(start, wall)| start + wall)
            .max()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_environment::records::{SessionId, Stamp};
    use cubecl_server::compiler::{CompilationOutcome, KernelCacheKey};

    /// A batch takes its wall, however much work its kernels did on their
    /// threads.
    #[test]
    fn a_batch_takes_its_wall_not_its_kernels_work() {
        let trips = [trip(0, 10), trip(0, 25), trip(0, 20), trip(100, 4)];
        let batches = [batch(0, 30), batch(100, 4)];
        let spans = CompilationSpans::new(&batches, &trips);
        assert_eq!(spans.within(&..), Duration::from_secs(34));
        assert_eq!(spans.end(), Duration::from_secs(104));
        assert_eq!(
            spans.within(&(Duration::ZERO..Duration::from_secs(50))),
            Duration::from_secs(30)
        );
    }

    /// A session that recorded no batch compiled each kernel alone.
    #[test]
    fn without_batches_each_kernel_takes_its_work() {
        let trips = [trip(0, 10), trip(10, 25)];
        let spans = CompilationSpans::new(&[], &trips);
        assert_eq!(spans.within(&..), Duration::from_secs(35));
        assert_eq!(spans.end(), Duration::from_secs(35));
    }

    fn stamp(start: u64) -> Stamp {
        Stamp {
            session: SessionId(0),
            seq: 0,
            offset: Duration::from_secs(start),
        }
    }

    fn batch(start: u64, wall: u64) -> Stamped<CompilationBatchRecord> {
        Stamped {
            stamp: stamp(start),
            record: CompilationBatchRecord {
                kernels: 1,
                threads: 1,
                wall: Duration::from_secs(wall),
            },
        }
    }

    fn trip(start: u64, work: u64) -> Stamped<CompilationRecord> {
        Stamped {
            stamp: stamp(start),
            record: CompilationRecord {
                kernel: "kernel".into(),
                key: KernelCacheKey { id: 0, build_id: 0 },
                ir: None,
                outcome: CompilationOutcome::Compiled,
                work: Duration::from_secs(work),
                source: None,
            },
        }
    }
}

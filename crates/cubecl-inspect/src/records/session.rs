use super::{CompilationSpans, EnvironmentRecords};
use crate::report::{Activity, SessionRow, SessionTimeline, SlowTune, Span, TuneTrace};
use cubecl_environment::records::{MarkRecord, Session, SessionId, Stamped};
use cubecl_server::compiler::CompilationRecord;
use std::ops::RangeBounds;
use std::time::Duration;

/// One session's share of the [`EnvironmentRecords`]: every list, filtered to
/// the records it stamped, and the stretches of its clock its compilations
/// took, laid out once for every report folded from them.
#[derive(Clone, Debug)]
pub struct SessionRecords<'a> {
    session: &'a Session,
    records: &'a EnvironmentRecords,
    compiling: CompilationSpans,
}

impl<'a> SessionRecords<'a> {
    /// The records of `records` that `session` stamped.
    pub fn new(session: &'a Session, records: &'a EnvironmentRecords) -> Self {
        Self {
            compiling: CompilationSpans::new(
                stamped_by(&records.batches, session.id),
                stamped_by(&records.compilations, session.id),
            ),
            session,
            records,
        }
    }

    pub fn session(&self) -> &'a Session {
        self.session
    }

    pub fn tunes(&self) -> impl Iterator<Item = &'a TuneTrace> {
        stamped_by(&self.records.tunes, self.session.id)
    }

    pub fn compilations(&self) -> impl Iterator<Item = &'a Stamped<CompilationRecord>> {
        stamped_by(&self.records.compilations, self.session.id)
    }

    pub fn marks(&self) -> impl Iterator<Item = &'a Stamped<MarkRecord>> {
        stamped_by(&self.records.marks, self.session.id)
    }

    /// The stretches of the session's clock its compilations took.
    pub fn compiling(&self) -> &CompilationSpans {
        &self.compiling
    }

    /// The tunes and compilations that started inside `window` of the
    /// session's clock, and what they took; `..` for the whole session.
    pub fn activity(&self, window: impl RangeBounds<Duration>) -> Activity {
        let mut activity = Activity {
            compilations: self
                .compilations()
                .filter(|trip| window.contains(&trip.stamp.offset))
                .count() as u64,
            compiling: self.compiling.within(&window),
            ..Activity::default()
        };
        for tune in self
            .tunes()
            .filter(|tune| window.contains(&tune.stamp.offset))
        {
            activity.tunes += 1;
            activity.tuning += tune.record.wall;
            activity.compiling_in_tunes += self.compiling.during(tune);
            if activity
                .slowest
                .as_ref()
                .is_none_or(|slowest| tune.record.wall > slowest.wall)
            {
                activity.slowest = Some(SlowTune {
                    table: tune.record.table.clone(),
                    key: tune.record.entry.key.clone(),
                    wall: tune.record.wall,
                });
            }
        }
        activity
    }
}

/// What the session's records add up to.
impl From<&SessionRecords<'_>> for SessionRow {
    fn from(records: &SessionRecords<'_>) -> Self {
        let tunes = records
            .tunes()
            .map(|trace| trace.stamp.offset + trace.record.wall);
        // A phase after the last compile — a walk served from the store —
        // still ran.
        let marks = records
            .marks()
            .map(|mark| mark.stamp.offset + mark.record.wall);
        let span = tunes
            .chain(marks)
            .chain([records.compiling.end()])
            .max()
            .unwrap_or_default();
        Self {
            session: records.session.clone(),
            activity: records.activity(..),
            span,
        }
    }
}

/// The session's marks laid out, each with what started inside it.
impl From<&SessionRecords<'_>> for SessionTimeline {
    fn from(records: &SessionRecords<'_>) -> Self {
        let mut spans: Vec<Span> = records
            .marks()
            .map(|mark| Span {
                stamp: mark.stamp,
                label: mark.record.label.clone(),
                wall: mark.record.wall,
                depth: 0,
                activity: Activity::default(),
            })
            .collect();
        // Opened first comes first; of two opened at once, the longer
        // encloses the other.
        spans.sort_by_key(|span| (span.stamp.offset, std::cmp::Reverse(span.wall)));
        for index in 0..spans.len() {
            let start = spans[index].stamp.offset;
            spans[index].depth = spans[..index]
                .iter()
                .filter(|outer| outer.window().contains(&start))
                .count();
            spans[index].activity = records.activity(spans[index].window());
        }
        Self {
            session: records.session.clone(),
            spans,
        }
    }
}

/// The records of `records` that `session` stamped.
fn stamped_by<R>(records: &[Stamped<R>], session: SessionId) -> impl Iterator<Item = &Stamped<R>> {
    records
        .iter()
        .filter(move |record| record.stamp.session == session)
}

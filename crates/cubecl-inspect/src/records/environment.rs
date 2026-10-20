use super::CompilationSpans;
use crate::report::{Activity, SlowTune, TuneTrace};
use cubecl_environment::persistence::Database;
use cubecl_environment::records::{MarkRecord, Records, Session, SessionId, Stamped};
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord};
use cubecl_server::memory_management::MemoryRecord;
use std::ops::RangeBounds;
use std::time::Duration;

/// Every record one environment file holds, each list oldest first: what
/// the reports of its sessions — their time, their kernels, their memory —
/// are folded from.
#[derive(Clone, Debug, Default)]
pub(crate) struct EnvironmentRecords {
    /// The processes that changed the file.
    pub(crate) sessions: Vec<Session>,
    pub(crate) tunes: Vec<TuneTrace>,
    pub(crate) compilations: Vec<Stamped<CompilationRecord>>,
    pub(crate) batches: Vec<Stamped<CompilationBatchRecord>>,
    pub(crate) marks: Vec<Stamped<MarkRecord>>,
    /// The memory snapshots the sessions took.
    pub(crate) memory: Vec<Stamped<MemoryRecord>>,
}

/// One session's share of the [`EnvironmentRecords`]: every list, filtered to the
/// records it stamped.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SessionRecords<'a> {
    session: &'a Session,
    records: &'a EnvironmentRecords,
}

impl EnvironmentRecords {
    /// The records of `database`, read whole.
    pub(crate) fn new(database: &Database) -> Self {
        let records = Records::new(database);
        Self {
            sessions: records.sessions(),
            tunes: records.read(),
            compilations: records.read(),
            batches: records.read(),
            marks: records.read(),
            memory: records.read(),
        }
    }

    /// Each session's records, oldest session first.
    pub(crate) fn sessions(&self) -> impl Iterator<Item = SessionRecords<'_>> {
        self.sessions.iter().map(|session| SessionRecords {
            session,
            records: self,
        })
    }

    /// The stretches of `session`'s clock its compilations took.
    pub(crate) fn compiling(&self, session: SessionId) -> CompilationSpans {
        CompilationSpans::new(
            stamped_by(&self.batches, session),
            stamped_by(&self.compilations, session),
        )
    }
}

impl<'a> SessionRecords<'a> {
    pub(crate) fn session(&self) -> &'a Session {
        self.session
    }

    pub(crate) fn tunes(&self) -> impl Iterator<Item = &'a TuneTrace> {
        stamped_by(&self.records.tunes, self.session.id)
    }

    pub(crate) fn compilations(&self) -> impl Iterator<Item = &'a Stamped<CompilationRecord>> {
        stamped_by(&self.records.compilations, self.session.id)
    }

    pub(crate) fn marks(&self) -> impl Iterator<Item = &'a Stamped<MarkRecord>> {
        stamped_by(&self.records.marks, self.session.id)
    }

    /// The stretches of the session's clock its compilations took.
    pub(crate) fn compiling(&self) -> CompilationSpans {
        self.records.compiling(self.session.id)
    }

    /// The tunes and compilations that started inside `window` of the
    /// session's clock, and what they took; `..` for the whole session.
    pub(crate) fn activity(&self, window: impl RangeBounds<Duration>) -> Activity {
        let compiling = self.compiling();
        let mut activity = Activity {
            compilations: self
                .compilations()
                .filter(|trip| window.contains(&trip.stamp.offset))
                .count() as u64,
            compiling: compiling.within(&window),
            ..Activity::default()
        };
        for tune in self
            .tunes()
            .filter(|tune| window.contains(&tune.stamp.offset))
        {
            activity.tunes += 1;
            activity.tuning += tune.record.wall;
            activity.compiling_in_tunes += compiling.during(tune);
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

/// The records of `records` that `session` stamped.
fn stamped_by<R>(records: &[Stamped<R>], session: SessionId) -> impl Iterator<Item = &Stamped<R>> {
    records
        .iter()
        .filter(move |record| record.stamp.session == session)
}

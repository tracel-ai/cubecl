use super::{CompilationSpans, RecordedTune, TuneTrace, TunedKey};
use cubecl_environment::records::{MarkRecord, Session, SessionId, Stamped};
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord};

/// Every record one environment file holds, each list oldest first: what
/// the reports that account for its sessions' time are folded from.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentRecords {
    /// The processes that changed the file.
    pub sessions: Vec<Session>,
    pub tunes: Vec<TuneTrace>,
    pub compilations: Vec<Stamped<CompilationRecord>>,
    pub batches: Vec<Stamped<CompilationBatchRecord>>,
    pub marks: Vec<Stamped<MarkRecord>>,
}

/// One session's share of the [`EnvironmentRecords`]: every list, filtered to the
/// records it stamped.
#[derive(Clone, Copy, Debug)]
pub struct SessionRecords<'a> {
    session: &'a Session,
    records: &'a EnvironmentRecords,
}

impl EnvironmentRecords {
    /// Each session's records, oldest session first.
    pub fn sessions(&self) -> impl Iterator<Item = SessionRecords<'_>> {
        self.sessions.iter().map(|session| SessionRecords {
            session,
            records: self,
        })
    }

    /// The last tune recorded of `key` that its table took, with the
    /// compiling inside it; `None` when there is none.
    pub fn recorded_tune(&self, key: &TunedKey) -> Option<RecordedTune> {
        let table = key.table.namespace();
        // Oldest first, so the last match is the key's latest tune. A tune the
        // table did not take answered its process alone, and says nothing of
        // the answer stored.
        let trace = self.tunes.iter().rev().find(|trace| {
            let record = &trace.record;
            record.stored
                && record.table == table
                && record.entry.checksum == key.checksum
                && record.entry.key == key.key
        })?;
        Some(RecordedTune {
            compiling: self.compiling(trace.stamp.session).during(trace),
            trace: trace.clone(),
        })
    }

    /// The stretches of `session`'s clock its compilations took.
    fn compiling(&self, session: SessionId) -> CompilationSpans {
        CompilationSpans::new(
            stamped_by(&self.batches, session),
            stamped_by(&self.compilations, session),
        )
    }
}

impl<'a> SessionRecords<'a> {
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
    pub fn compiling(&self) -> CompilationSpans {
        self.records.compiling(self.session.id)
    }
}

/// The records of `records` that `session` stamped.
fn stamped_by<R>(records: &[Stamped<R>], session: SessionId) -> impl Iterator<Item = &Stamped<R>> {
    records
        .iter()
        .filter(move |record| record.stamp.session == session)
}

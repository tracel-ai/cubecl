use super::{CanonicalKey, CompilationSpans, RecordedTune, TuneTrace, TunedKey};
use cubecl_environment::records::{MarkRecord, Session, SessionId, Stamped};
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord};
use cubecl_server::memory_management::MemoryRecord;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Every record one environment file holds, each list oldest first: what
/// the reports of its sessions — their time, their kernels, their memory —
/// are folded from.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentRecords {
    /// The processes that changed the file.
    pub sessions: Vec<Session>,
    pub tunes: Vec<TuneTrace>,
    pub compilations: Vec<Stamped<CompilationRecord>>,
    pub batches: Vec<Stamped<CompilationBatchRecord>>,
    pub marks: Vec<Stamped<MarkRecord>>,
    /// The memory snapshots the sessions took.
    pub memory: Vec<Stamped<MemoryRecord>>,
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

    /// The stretches of `session`'s clock its compilations took.
    fn compiling(&self, session: SessionId) -> CompilationSpans {
        CompilationSpans::new(
            stamped_by(&self.batches, session),
            stamped_by(&self.compilations, session),
        )
    }
}

/// The tunes of a file that its tables took, by the key each answers, with
/// the compiling of every session that ran one: built once, so joining each
/// key to its tune is a lookup rather than a scan of every record.
pub struct TuneIndex<'a> {
    /// A key's latest stored tune, by its table's namespace, its checksum and
    /// the key itself.
    tunes: HashMap<(&'a str, &'a str, CanonicalKey), &'a TuneTrace>,
    compiling: BTreeMap<SessionId, CompilationSpans>,
}

impl<'a> TuneIndex<'a> {
    pub fn new(records: &'a EnvironmentRecords) -> Self {
        let mut tunes = HashMap::new();
        // Oldest first, so a later tune of a key replaces the earlier: the
        // index keeps each key's latest. A tune the table did not take
        // answered its process alone, and says nothing of the answer stored.
        for trace in records.tunes.iter().filter(|trace| trace.record.stored) {
            let record = &trace.record;
            let entry = (
                record.table.as_str(),
                record.entry.checksum.as_str(),
                CanonicalKey::new(&record.entry.key),
            );
            tunes.insert(entry, trace);
        }
        let sessions: BTreeSet<SessionId> =
            tunes.values().map(|trace| trace.stamp.session).collect();
        let compiling = sessions
            .into_iter()
            .map(|session| (session, records.compiling(session)))
            .collect();
        Self { tunes, compiling }
    }

    /// The last tune recorded of `key` that its table took, with the
    /// compiling inside it; `None` when there is none.
    pub fn recorded_tune(&self, key: &TunedKey) -> Option<RecordedTune> {
        let table = key.table.namespace();
        let entry = (
            table.as_str(),
            key.checksum.as_str(),
            CanonicalKey::new(&key.key),
        );
        let trace = *self.tunes.get(&entry)?;
        Some(RecordedTune {
            compiling: self.compiling[&trace.stamp.session].during(trace),
            trace: trace.clone(),
        })
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

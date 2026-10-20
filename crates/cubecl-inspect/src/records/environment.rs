use super::{CompilationSpans, SessionRecords};
use crate::report::UntypedTuneRecord;
use cubecl_environment::persistence::Database;
use cubecl_environment::records::{MarkRecord, Records, Session, SessionId, Stamped};
use cubecl_server::compiler::{CompilationBatchRecord, CompilationRecord};
use cubecl_server::memory_management::MemoryRecord;

/// Every record one environment file holds, each list oldest first: what
/// the reports of its sessions — their time, their kernels, their memory —
/// are folded from.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentRecords {
    /// The processes that changed the file.
    pub sessions: Vec<Session>,
    pub tunes: Vec<Stamped<UntypedTuneRecord>>,
    pub compilations: Vec<Stamped<CompilationRecord>>,
    pub batches: Vec<Stamped<CompilationBatchRecord>>,
    pub marks: Vec<Stamped<MarkRecord>>,
    /// The memory snapshots the sessions took.
    pub memory: Vec<Stamped<MemoryRecord>>,
}

impl EnvironmentRecords {
    /// The records of `database`, read whole.
    pub fn new(database: &Database) -> Self {
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
    pub fn sessions(&self) -> impl Iterator<Item = SessionRecords<'_>> {
        self.sessions
            .iter()
            .map(|session| SessionRecords::new(session, self))
    }

    /// The stretches of `session`'s clock its compilations took: read by its
    /// id, so a session whose row was pruned while it ran still has them.
    pub fn compiling(&self, session: SessionId) -> CompilationSpans {
        CompilationSpans::new(
            self.batches
                .iter()
                .filter(|batch| batch.stamp.session == session),
            self.compilations
                .iter()
                .filter(|trip| trip.stamp.session == session),
        )
    }
}

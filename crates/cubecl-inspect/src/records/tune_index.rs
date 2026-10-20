use super::{CompilationSpans, EnvironmentRecords, StampedRecords};
use crate::report::{CanonicalKey, RecordedTune, TunedKey, UntypedTuneRecord};
use cubecl_environment::records::{SessionId, Stamped};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The tunes of a file that its tables took, by the key each answers, with
/// the compiling of every session that ran one: built once, so joining each
/// key to its tune is a lookup rather than a scan of every record.
pub struct TuneIndex<'a> {
    /// A key's latest stored tune, by its table's namespace, its checksum and
    /// the key itself.
    tunes: HashMap<(&'a str, &'a str, CanonicalKey), &'a Stamped<UntypedTuneRecord>>,
    /// The compiling of each session that ran one.
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
        let tuned: BTreeSet<SessionId> = tunes.values().map(|trace| trace.stamp.session).collect();
        let compiling = tuned
            .into_iter()
            .map(|session| (session, StampedRecords::new(records, session).compiling))
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

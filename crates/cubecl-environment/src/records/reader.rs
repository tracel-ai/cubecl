use super::{ROOT, Record, SESSIONS, Session, SessionId, Stamped};
use crate::bytes::Bytes;
use crate::environment::{PruneTally, RecordPruning};
use crate::persistence::{Database, NamespaceContent, NamespaceRoot, encode};
use alloc::vec::Vec;

/// The records and sessions one database holds.
#[derive(Debug, Clone, Copy)]
pub struct Records<'a> {
    database: &'a Database,
}

impl<'a> Records<'a> {
    /// The records `database` holds.
    pub fn new(database: &'a Database) -> Self {
        Self { database }
    }

    /// Every session, oldest first.
    pub fn sessions(&self) -> Vec<Session> {
        let mut sessions = Vec::new();
        self.database.scan(SESSIONS, &mut |_, value| {
            if let Ok(session) = ciborium::from_reader::<Session, _>(value) {
                sessions.push(session);
            }
        });
        sessions.sort_by_key(|session| session.id);
        sessions
    }

    /// Every record of `R` that decodes, in the order they were stamped.
    pub fn read<R: Record + serde::de::DeserializeOwned>(&self) -> Vec<Stamped<R>> {
        let mut records = Vec::new();
        self.database.scan(&R::namespace(), &mut |_, value| {
            if let Ok(record) = ciborium::from_reader::<Stamped<R>, _>(value) {
                records.push(record);
            }
        });
        records.sort_by_key(|record: &Stamped<R>| (record.stamp.session, record.stamp.seq));
        records
    }

    /// Keep the records `pruning` keeps and delete the rest, counting the
    /// entries kept and deleted. A layout older than this one has no
    /// sessions this reader knows: removing every record removes it too, and
    /// keeping the latest sessions keeps it.
    pub fn retain(&self, pruning: RecordPruning) -> PruneTally {
        let before = self.entries();
        match pruning {
            RecordPruning::RemoveAll => self.remove_all(),
            RecordPruning::KeepLatest { sessions } => {
                self.prune(sessions as usize);
            }
            RecordPruning::KeepAll => {}
        }
        let after = self.entries();
        // Another process recording into the environment meanwhile can leave
        // more than there was: what it added was never removed.
        PruneTally {
            kept: after.entries,
            removed: before.entries.saturating_sub(after.entries),
            removed_bytes: before.bytes.saturating_sub(after.bytes),
        }
    }

    /// Deletes every session and every record, of every layout.
    fn remove_all(&self) {
        for namespace in self.database.namespaces() {
            if holds_records(&namespace) {
                self.database.purge(&namespace);
            }
        }
    }

    /// The entries of every layout of records, and the bytes they take.
    fn entries(&self) -> RecordSize {
        let mut total = RecordSize::default();
        // A count the database refuses tallies nothing; the database logs why.
        for summary in self.database.summary().unwrap_or_default() {
            if holds_records(&summary.namespace) {
                total.entries += summary.entries;
                total.bytes += summary.bytes;
            }
        }
        total
    }

    /// Deletes every session but the newest `keep`, with their records.
    /// Returns how many sessions went.
    ///
    /// Every record stamped at or before the newest session pruned goes too,
    /// whether or not its session is still listed: a process whose session
    /// another one pruned while it ran keeps writing under it, and those
    /// records go at the next prune rather than never.
    pub fn prune(&self, keep: usize) -> usize {
        let mut sessions = self.sessions();
        if sessions.len() <= keep {
            return 0;
        }
        sessions.sort_by_key(|session| core::cmp::Reverse(session.id));
        let pruned = &sessions[keep..];
        let newest_pruned = pruned[0].id;

        for namespace in self.database.namespaces() {
            if namespace == SESSIONS || !namespace.starts_with(ROOT) {
                continue;
            }
            let mut doomed = Vec::new();
            self.database.scan(&namespace, &mut |key, _| {
                if let Ok((session, _)) = ciborium::from_reader::<(SessionId, u64), _>(key)
                    && session <= newest_pruned
                {
                    doomed.push(Bytes::from_bytes_vec(key.to_vec()));
                }
            });
            for key in doomed {
                self.database.purge_key(&namespace, &key);
            }
        }
        for session in pruned {
            self.database.purge_key(SESSIONS, &encode(&session.id));
        }
        pruned.len()
    }
}

/// How many entries the records hold, and the bytes of their keys and values.
#[derive(Default)]
struct RecordSize {
    entries: u64,
    bytes: u64,
}

/// Whether `namespace` holds records or sessions, of this layout or an earlier
/// one.
fn holds_records(namespace: &str) -> bool {
    NamespaceRoot::new(namespace).content() == NamespaceContent::Records
}

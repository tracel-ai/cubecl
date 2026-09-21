use super::{ROOT, Record, SESSIONS, Session, SessionId, Stamped};
use crate::bytes::Bytes;
use crate::persistence::{Database, encode};
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

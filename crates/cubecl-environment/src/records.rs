//! What an environment remembers of how it was built.
//!
//! The caches say *what* was decided — a winner per autotune key, a binary per
//! kernel. Records say *how*: which candidates a tune ran and in what order,
//! what each cost, when it happened. They live in the environment they
//! describe, under [`ROOT`], so a file carries the account of its own build
//! and a reader needs nothing else to explain it.
//!
//! Every record belongs to a [`Session`] — one process's use of one
//! environment — and carries a [`Stamp`]: the session, a sequence number, and
//! an offset from the session's start. That is what restores order across
//! subsystems that write independently, and lays them on one time axis.
//!
//! Recording is a write on a path that already writes to the environment (a
//! tune, a compile) or once per session: nothing here runs per launch. The
//! [`RecordLevel`] turns it off, or on in full, where a record can carry
//! something heavy.

use alloc::string::String;
#[cfg(native_cache)]
use alloc::string::ToString;
use core::time::Duration;
use serde::{Deserialize, Serialize};

/// The namespace root of every record, versioned so a reader selects the
/// layout it knows and an export can drop them all by one prefix.
pub const ROOT: &str = "records/v1";

/// The namespace sessions are written to.
pub const SESSIONS: &str = "records/v1/sessions";

/// How much an environment records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum RecordLevel {
    /// Nothing.
    Off = 0,
    /// Every record, without the heavy parts: what a build cost, not the
    /// artifacts it produced.
    #[default]
    Basic = 1,
    /// Every record in full: kernel sources, allocation histograms.
    Full = 2,
}

/// One process's use of one environment: what every record of it shares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Unique among the sessions of an environment.
    pub id: u64,
    /// Wall-clock start, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// The cubecl that wrote it.
    pub cubecl_version: String,
    /// What the process said it was doing, e.g. `models build qwen3-8b`.
    pub label: Option<String>,
    /// The operating system's process id.
    pub process: u32,
    /// The operating system, e.g. `linux`.
    pub os: String,
    /// The CPU architecture, e.g. `x86_64`.
    pub arch: String,
}

/// Where a record sits in its session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    /// The [`Session::id`] of the session the record belongs to.
    pub session: u64,
    /// The record's position among its session's, across every namespace.
    pub seq: u64,
    /// Time since the session started, on a monotonic clock.
    pub offset: Duration,
}

/// A record as it is stored: its stamp, then its body. The key is the stamp's
/// `(session, seq)`, which is unique and sorts a session's records in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stamped<V> {
    /// Where the record sits in its session.
    pub stamp: Stamp,
    /// The record itself.
    pub record: V,
}

/// Sets how much is recorded, and how many sessions an environment keeps: at
/// the start of a session, the oldest beyond `keep_sessions` are pruned with
/// their records. Called by the runtime once its configuration is loaded.
pub fn configure(level: RecordLevel, keep_sessions: Option<u32>) {
    imp::configure(level, keep_sessions);
}

/// The level in force.
pub fn level() -> RecordLevel {
    imp::level()
}

/// Whether anything is recorded.
pub fn enabled() -> bool {
    level() != RecordLevel::Off
}

/// Names what this process is doing, for the session in progress and every
/// later one: `models build qwen3-8b`, `serve`.
pub fn label<S: Into<String>>(label: S) {
    imp::label(label.into());
}

/// Writes `record` under `records/v1/<kind>` in the active environment,
/// stamped now. `None` when recording is off or there is no database to write
/// to.
pub fn write<V: Serialize>(kind: &str, record: &V) -> Option<Stamp> {
    let stamp = stamp()?;
    write_stamped(kind, stamp, record).then_some(stamp)
}

/// Writes `record` with a stamp taken earlier, for a record that describes a
/// span: stamped when it began, written when it ended.
pub fn write_stamped<V: Serialize>(kind: &str, stamp: Stamp, record: &V) -> bool {
    imp::write(kind, &Stamped { stamp, record })
}

/// A stamp for this moment in the active environment's session, starting the
/// session if there is none. `None` when recording is off.
pub fn stamp() -> Option<Stamp> {
    imp::stamp()
}

/// The namespace records of `kind` are written to.
pub fn namespace(kind: &str) -> String {
    alloc::format!("{ROOT}/{}", kind.trim_matches('/'))
}

/// Opens a named span of the session in progress — a phase of the caller's
/// work, `load weights`, `plan turn 7` — recorded as a [`MarkRecord`] when the
/// returned guard drops. Everything else stamped meanwhile lies inside it on
/// the session's clock, which is what lets a reader say what a phase cost.
pub fn mark<S: Into<String>>(label: S) -> Mark {
    Mark {
        label: label.into(),
        start: stamp(),
    }
}

/// A span opened by [`mark`], recorded when it drops.
#[must_use = "a mark records the span until it is dropped"]
#[derive(Debug)]
pub struct Mark {
    label: String,
    start: Option<Stamp>,
}

/// A named span of a session, as [`mark`] records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkRecord {
    /// What the span was.
    pub label: String,
    /// How long it lasted.
    pub wall: Duration,
}

impl MarkRecord {
    /// The records namespace kind marks are written under.
    pub const KIND: &str = "marks";
}

impl Drop for Mark {
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        // The end is read off the session's own clock. A span the
        // environment switched away from in the meantime is dropped with its
        // session.
        let Some(end) = stamp().filter(|end| end.session == start.session) else {
            return;
        };
        let record = MarkRecord {
            label: core::mem::take(&mut self.label),
            wall: end.offset.saturating_sub(start.offset),
        };
        write_stamped(MarkRecord::KIND, start, &record);
    }
}

/// Deletes every session of `database` but the newest `keep`, with their
/// records. Returns how many sessions went.
#[cfg(native_cache)]
pub fn prune(database: &crate::persistence::Database, keep: usize) -> usize {
    imp::prune(database, keep)
}

/// Every session `database` holds, oldest first.
#[cfg(native_cache)]
pub fn sessions(database: &crate::persistence::Database) -> alloc::vec::Vec<Session> {
    let mut sessions = alloc::vec::Vec::new();
    database.scan(SESSIONS, &mut |_, value| {
        if let Ok(session) = ciborium::from_reader::<Session, _>(value) {
            sessions.push(session);
        }
    });
    sessions.sort_by_key(|session| session.id);
    sessions
}

/// Every record of `kind` in `database` that decodes as `V`, in the order
/// they were stamped.
#[cfg(native_cache)]
pub fn read<V: serde::de::DeserializeOwned>(
    database: &crate::persistence::Database,
    kind: &str,
) -> alloc::vec::Vec<Stamped<V>> {
    let mut records = alloc::vec::Vec::new();
    database.scan(&namespace(kind), &mut |_, value| {
        if let Ok(record) = ciborium::from_reader::<Stamped<V>, _>(value) {
            records.push(record);
        }
    });
    records.sort_by_key(|record: &Stamped<V>| (record.stamp.session, record.stamp.seq));
    records
}

#[cfg(native_cache)]
impl Session {
    fn new(label: Option<String>) -> Self {
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            // Nanoseconds are unique enough between the processes sharing one
            // environment, and sort sessions by start without a lookup.
            id: started.as_nanos() as u64,
            started_unix_ms: started.as_millis() as u64,
            cubecl_version: env!("CARGO_PKG_VERSION").to_string(),
            label,
            process: std::process::id(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }
}

/// Recording where there is a database to record into.
#[cfg(native_cache)]
mod imp {
    use super::{RecordLevel, SESSIONS, Session, Stamp};
    use crate::persistence::{Database, Origin};
    use crate::sync::{AtomicU8, LazyLock, Mutex, Ordering};
    use alloc::string::String;
    use alloc::vec::Vec;
    use serde::Serialize;
    use std::time::Instant;

    /// The level, apart from the rest of the state: [`super::enabled`] is
    /// asked on paths that must not take a lock.
    static LEVEL: AtomicU8 = AtomicU8::new(RecordLevel::Basic as u8);

    struct State {
        keep_sessions: Option<u32>,
        label: Option<String>,
        current: Option<Current>,
    }

    /// The session in progress, bound to the environment it was started in.
    struct Current {
        session: Session,
        database: Database,
        generation: u32,
        started: Instant,
        next_seq: u64,
    }

    static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
        Mutex::new(State {
            keep_sessions: None,
            label: None,
            current: None,
        })
    });

    pub(super) fn configure(level: RecordLevel, keep_sessions: Option<u32>) {
        LEVEL.store(level as u8, Ordering::Relaxed);
        STATE.lock().keep_sessions = keep_sessions;
    }

    pub(super) fn level() -> RecordLevel {
        match LEVEL.load(Ordering::Relaxed) {
            0 => RecordLevel::Off,
            1 => RecordLevel::Basic,
            _ => RecordLevel::Full,
        }
    }

    pub(super) fn label(label: String) {
        let mut state = STATE.lock();
        state.label = Some(label.clone());
        if let Some(current) = state.current.as_mut() {
            current.session.label = Some(label);
            write_session(current);
        }
    }

    pub(super) fn stamp() -> Option<Stamp> {
        let mut state = STATE.lock();
        let current = current(&mut state)?;
        let stamp = Stamp {
            session: current.session.id,
            seq: current.next_seq,
            offset: current.started.elapsed(),
        };
        current.next_seq += 1;
        Some(stamp)
    }

    pub(super) fn write<V: Serialize>(kind: &str, stamped: &super::Stamped<V>) -> bool {
        let database = {
            let mut state = STATE.lock();
            match current(&mut state) {
                // A record stamped in a session the environment has since
                // switched away from belongs to neither.
                Some(current) if current.session.id == stamped.stamp.session => {
                    current.database.clone()
                }
                _ => return false,
            }
        };
        let key = encode(&(stamped.stamp.session, stamped.stamp.seq));
        let value = encode(stamped);
        matches!(
            database.insert(&super::namespace(kind), &key, &value, Origin::Local),
            crate::persistence::Insertion::Stored
        )
    }

    /// The session of the active environment, started if there is none or the
    /// environment switched since. `None` when recording is off or the
    /// environment has no database.
    fn current(state: &mut State) -> Option<&mut Current> {
        if level() == RecordLevel::Off {
            return None;
        }
        let generation = crate::environment::generation();
        let stale = state
            .current
            .as_ref()
            .is_none_or(|current| current.generation != generation);
        if stale {
            let database = Database::open_active()?;
            if let Some(keep) = state.keep_sessions {
                // Room for the one about to start.
                prune(&database, keep.saturating_sub(1) as usize);
            }
            let current = Current {
                session: Session::new(state.label.clone()),
                database,
                generation,
                started: Instant::now(),
                next_seq: 0,
            };
            write_session(&current);
            state.current = Some(current);
        }
        state.current.as_mut()
    }

    fn write_session(current: &Current) {
        let key = encode(&current.session.id);
        current
            .database
            .replace(SESSIONS, &key, &encode(&current.session), Origin::Local);
    }

    pub(super) fn prune(database: &Database, keep: usize) -> usize {
        let mut sessions = super::sessions(database);
        if sessions.len() <= keep {
            return 0;
        }
        sessions.sort_by_key(|session| core::cmp::Reverse(session.id));
        let pruned: Vec<u64> = sessions[keep..].iter().map(|session| session.id).collect();

        for namespace in database.namespaces() {
            if namespace == SESSIONS || !namespace.starts_with(super::ROOT) {
                continue;
            }
            let mut doomed = Vec::new();
            database.scan(&namespace, &mut |key, _| {
                if let Ok((session, _)) = ciborium::from_reader::<(u64, u64), _>(key)
                    && pruned.contains(&session)
                {
                    doomed.push(key.to_vec());
                }
            });
            for key in doomed {
                database.purge_key(&namespace, &key);
            }
        }
        for id in &pruned {
            database.purge_key(SESSIONS, &encode(id));
        }
        pruned.len()
    }

    fn encode<V: Serialize + ?Sized>(value: &V) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(value, &mut bytes).expect("a record serializes");
        bytes
    }
}

/// Nowhere to record into: every call is a no-op and nothing is ever stamped.
#[cfg(not(native_cache))]
mod imp {
    use super::{RecordLevel, Stamp};
    use alloc::string::String;

    pub(super) fn configure(_level: RecordLevel, _keep_sessions: Option<u32>) {}

    pub(super) fn level() -> RecordLevel {
        RecordLevel::Off
    }

    pub(super) fn label(_label: String) {}

    pub(super) fn stamp() -> Option<Stamp> {
        None
    }

    pub(super) fn write<V: serde::Serialize>(_kind: &str, _stamped: &super::Stamped<V>) -> bool {
        false
    }
}

#[cfg(all(test, native_cache))]
mod tests {
    use super::*;
    use crate::persistence::Database;
    use alloc::vec;
    use alloc::vec::Vec;
    use serial_test::serial;

    /// Points the active environment at a fresh database, recording at
    /// `level`, and hands back the database to read.
    fn fresh(level: RecordLevel, keep: Option<u32>) -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().expect("a temp dir");
        crate::environment::set_root(dir.path());
        crate::environment::activate(alloc::format!("records-{}", std::process::id()));
        configure(level, keep);
        let database = Database::open_active().expect("opens");
        (dir, database)
    }

    fn records(database: &Database, kind: &str) -> Vec<Stamped<u32>> {
        read(database, kind)
    }

    #[test]
    #[serial]
    fn records_are_stamped_in_order_within_one_session() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let first = write("test", &1u32).expect("recorded");
        let second = write("other", &2u32).expect("recorded");

        assert_eq!(first.session, second.session);
        assert_eq!(second.seq, first.seq + 1);
        assert!(second.offset >= first.offset);
        assert_eq!(records(&database, "test")[0].record, 1);
        assert_eq!(records(&database, "other")[0].stamp, second);
    }

    #[test]
    #[serial]
    fn a_session_is_written_once_and_carries_its_label() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        label("models build test");
        write("test", &1u32).expect("recorded");

        let sessions = sessions(&database);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].label.as_deref(), Some("models build test"));
    }

    #[test]
    #[serial]
    fn a_mark_spans_what_was_recorded_inside_it() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let inner = {
            let _phase = mark("load weights");
            write("test", &1u32).expect("recorded")
        };

        let marks = read::<MarkRecord>(&database, MarkRecord::KIND);
        assert_eq!(marks.len(), 1);
        let span = &marks[0];
        assert_eq!(span.record.label, "load weights");
        assert!(span.stamp.seq < inner.seq);
        assert!(span.stamp.offset + span.record.wall >= inner.offset);
    }

    #[test]
    #[serial]
    fn nothing_is_recorded_when_off() {
        let (_dir, database) = fresh(RecordLevel::Off, None);
        assert_eq!(write("test", &1u32), None);
        assert!(database.namespaces().is_empty());
    }

    #[test]
    #[serial]
    fn a_switch_starts_a_new_session_and_pruning_keeps_the_newest() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let first = write("test", &1u32).expect("recorded");
        // Re-activating is a switch: the same database, a new session.
        crate::environment::activate(crate::environment::active());
        let second = write("test", &2u32).expect("recorded");
        assert_ne!(first.session, second.session);
        assert_eq!(records(&database, "test").len(), 2);

        assert_eq!(prune(&database, 1), 1);
        let kept = records(&database, "test");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].stamp.session, second.session);
    }
}

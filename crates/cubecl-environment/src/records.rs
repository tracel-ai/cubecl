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
//! A type is written as a record by implementing [`Record`], and read back
//! through [`Records`].
//!
//! A session is kept only if it changed the environment — tuned a key,
//! compiled a kernel. Until its first such record, what it records is held in
//! memory, up to a budget past which the oldest goes first; a session that
//! never changes anything — a warm-up that finds every kernel stored and every
//! key tuned, a server that runs for days on a warm environment — leaves
//! nothing behind and holds a bounded amount. Each record says which it is
//! with its [`RecordEffect`].
//!
//! A record that describes a stretch of time — a tune, a compile, a phase of
//! the caller's work — is written through a [`Span`]: stamped when it opens,
//! timed on the session's clock, written when it closes.
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
    /// Every record in full: a compiled kernel's IR and source.
    Full = 2,
}

/// Whether a record accompanies a change to the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordEffect {
    /// The environment's caches changed: a key was tuned, a kernel compiled.
    /// The session is kept, with everything it recorded before.
    Changed,
    /// Something happened that changed nothing: a kernel loaded from the
    /// store, a span marked, a snapshot taken. Kept only if the session
    /// changes something.
    Observed,
}

/// A type written to the environment as a record: its kind names the
/// namespace, `records/v1/<KIND>`, it is written to and read back from.
pub trait Record {
    /// The kind records of this type are written under.
    const KIND: &'static str;

    /// The namespace records of this type are written to.
    fn namespace() -> String {
        alloc::format!("{ROOT}/{}", Self::KIND.trim_matches('/'))
    }
}

/// Names a [`Session`]: unique among the sessions of an environment. Taken
/// from the session's start in nanoseconds, so ids sort sessions by start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub u64);

impl core::fmt::Display for SessionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One process's use of one environment: what every record of it shares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Unique among the sessions of an environment.
    pub id: SessionId,
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
    pub session: SessionId,
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

/// Sets how much is recorded, and how many sessions an environment keeps:
/// when a session is kept, the oldest beyond `keep_sessions` are pruned with
/// their records; `None` keeps them all. The session being kept is never
/// pruned, so `Some(0)` keeps it alone, as `Some(1)` does. Called by the
/// runtime once its configuration is loaded.
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

/// Records `record` under its [namespace](Record::namespace) in the active
/// environment, stamped now: written if the session is kept, which an
/// [`Observed`](RecordEffect::Observed) record alone does not decide. `None`
/// when recording is off or there is no database to write to.
pub fn write<R: Record + Serialize>(effect: RecordEffect, record: &R) -> Option<Stamp> {
    let stamp = imp::stamp()?;
    write_stamped(stamp, effect, record).then_some(stamp)
}

/// Records `record` with a stamp taken earlier. `false` when the session it
/// was stamped in is gone, or the write failed.
fn write_stamped<R: Record + Serialize>(stamp: Stamp, effect: RecordEffect, record: &R) -> bool {
    imp::write(&R::namespace(), effect, &Stamped { stamp, record })
}

/// A stretch of the session in progress that a record describes: stamped when
/// it opens, timed on the session's clock, and recorded when it closes, so
/// everything stamped meanwhile lies inside it.
#[must_use = "a span records nothing until it is closed"]
#[derive(Debug)]
pub struct Span {
    start: Stamp,
}

impl Span {
    /// Opens a span now, starting the session if there is none. `None` when
    /// recording is off.
    pub fn new() -> Option<Self> {
        imp::stamp().map(|start| Self { start })
    }

    /// The time since the span opened, on the session's clock. `None` when
    /// the environment switched sessions since: the span went with its own.
    pub fn elapsed(&self) -> Option<Duration> {
        imp::offset(self.start.session).map(|now| now.saturating_sub(self.start.offset))
    }

    /// Records `record`, stamped when the span opened: written if the session
    /// is kept, as [`write()`] is. `false` when the session the span opened in
    /// is gone, or the write failed.
    pub fn close<R: Record + Serialize>(self, effect: RecordEffect, record: &R) -> bool {
        write_stamped(self.start, effect, record)
    }
}

/// Opens a named span of the session in progress — a phase of the caller's
/// work, `load weights`, `plan turn 7` — recorded as a [`MarkRecord`] when the
/// returned guard drops. Everything else stamped meanwhile lies inside it on
/// the session's clock, which is what lets a reader say what a phase cost.
pub fn mark<S: Into<String>>(label: S) -> Mark {
    Mark {
        label: label.into(),
        span: Span::new(),
    }
}

/// A phase opened by [`mark`], recorded when it drops.
#[must_use = "a mark records the span until it is dropped"]
#[derive(Debug)]
pub struct Mark {
    label: String,
    span: Option<Span>,
}

/// A named span of a session, as [`mark`] records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkRecord {
    /// What the span was.
    pub label: String,
    /// How long it lasted.
    pub wall: Duration,
}

impl Record for MarkRecord {
    const KIND: &'static str = "marks";
}

impl Drop for Mark {
    fn drop(&mut self) {
        let Some(span) = self.span.take() else {
            return;
        };
        let Some(wall) = span.elapsed() else {
            return;
        };
        let record = MarkRecord {
            label: core::mem::take(&mut self.label),
            wall,
        };
        span.close(RecordEffect::Observed, &record);
    }
}

/// The records and sessions one database holds.
#[cfg(native_cache)]
#[derive(Debug, Clone, Copy)]
pub struct Records<'a> {
    database: &'a crate::persistence::Database,
}

#[cfg(native_cache)]
impl<'a> Records<'a> {
    /// The records `database` holds.
    pub fn new(database: &'a crate::persistence::Database) -> Self {
        Self { database }
    }

    /// Every session, oldest first.
    pub fn sessions(&self) -> alloc::vec::Vec<Session> {
        let mut sessions = alloc::vec::Vec::new();
        self.database.scan(SESSIONS, &mut |_, value| {
            if let Ok(session) = ciborium::from_reader::<Session, _>(value) {
                sessions.push(session);
            }
        });
        sessions.sort_by_key(|session| session.id);
        sessions
    }

    /// Every record of `R` that decodes, in the order they were stamped.
    pub fn read<R: Record + serde::de::DeserializeOwned>(&self) -> alloc::vec::Vec<Stamped<R>> {
        let mut records = alloc::vec::Vec::new();
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
            let mut doomed = alloc::vec::Vec::new();
            self.database.scan(&namespace, &mut |key, _| {
                if let Ok((session, _)) = ciborium::from_reader::<(SessionId, u64), _>(key)
                    && session <= newest_pruned
                {
                    doomed.push(key.to_vec());
                }
            });
            for key in doomed {
                self.database.purge_key(&namespace, &key);
            }
        }
        for session in pruned {
            self.database.purge_key(SESSIONS, &imp::encode(&session.id));
        }
        pruned.len()
    }
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
            id: SessionId(started.as_nanos() as u64),
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
    use super::{RecordEffect, RecordLevel, Records, SESSIONS, Session, SessionId, Stamp};
    use crate::persistence::{Database, Origin};
    use crate::sync::{AtomicU8, LazyLock, Mutex, Ordering};
    use alloc::collections::VecDeque;
    use alloc::string::String;
    use alloc::vec::Vec;
    use serde::Serialize;
    use std::time::Instant;

    /// The level, apart from the rest of the state: [`super::enabled`] is
    /// asked on paths that must not take a lock.
    static LEVEL: AtomicU8 = AtomicU8::new(RecordLevel::Basic as u8);

    /// How many encoded bytes a session holds before it changes anything.
    /// Past it the oldest record goes first: what led up to a change is kept,
    /// and a session that never changes anything cannot grow without bound.
    const PENDING_BUDGET: usize = 1 << 20;

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
        /// What the session recorded before it changed anything, in order:
        /// written if it does, dropped with it if it never does. Empty once
        /// the session is kept, and never past [`PENDING_BUDGET`].
        pending: VecDeque<Row>,
        /// The encoded size of [`Self::pending`].
        pending_bytes: usize,
        /// Whether the session changed the environment, and so is written.
        kept: bool,
    }

    /// One encoded record: its namespace, key and value.
    struct Row {
        namespace: String,
        key: Vec<u8>,
        value: Vec<u8>,
    }

    impl Row {
        fn bytes(&self) -> usize {
            self.namespace.len() + self.key.len() + self.value.len()
        }
    }

    impl Current {
        /// Hold `row` until the session changes something, dropping the
        /// oldest rows held past the budget.
        fn hold(&mut self, row: Row) {
            self.pending_bytes += row.bytes();
            self.pending.push_back(row);
            while self.pending_bytes > PENDING_BUDGET {
                let Some(oldest) = self.pending.pop_front() else {
                    break;
                };
                self.pending_bytes -= oldest.bytes();
            }
        }
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
            if current.kept {
                write_session(current);
            }
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

    /// The session clock's reading, when `session` is still the one in
    /// progress.
    pub(super) fn offset(session: SessionId) -> Option<core::time::Duration> {
        let state = STATE.lock();
        let current = state.current.as_ref()?;
        let live =
            current.session.id == session && current.generation == crate::environment::generation();
        live.then(|| current.started.elapsed())
    }

    pub(super) fn write<V: Serialize>(
        namespace: &str,
        effect: RecordEffect,
        stamped: &super::Stamped<V>,
    ) -> bool {
        let row = Row {
            namespace: namespace.into(),
            key: encode(&(stamped.stamp.session, stamped.stamp.seq)),
            value: encode(stamped),
        };
        let mut state = STATE.lock();
        let keep_sessions = state.keep_sessions;
        let Some(current) = current(&mut state) else {
            return false;
        };
        // A record stamped in a session the environment has since switched
        // away from belongs to neither.
        if current.session.id != stamped.stamp.session {
            return false;
        }
        match (current.kept, effect) {
            (true, _) => insert(&current.database, &row),
            (false, RecordEffect::Observed) => {
                current.hold(row);
                true
            }
            (false, RecordEffect::Changed) => {
                keep(current, keep_sessions);
                insert(&current.database, &row)
            }
        }
    }

    /// The session changed the environment: write it, with what it recorded
    /// before, and make room for it among the sessions kept.
    fn keep(current: &mut Current, keep_sessions: Option<u32>) {
        if let Some(keep) = keep_sessions {
            Records::new(&current.database).prune(keep.saturating_sub(1) as usize);
        }
        current.kept = true;
        write_session(current);
        current.pending_bytes = 0;
        for row in core::mem::take(&mut current.pending) {
            insert(&current.database, &row);
        }
    }

    fn insert(database: &Database, row: &Row) -> bool {
        matches!(
            database.insert(&row.namespace, &row.key, &row.value, Origin::Local),
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
            // A session replaced before it changed anything drops what it
            // was holding with it.
            state.current = Some(Current {
                session: Session::new(state.label.clone()),
                database: Database::open_active()?,
                generation,
                started: Instant::now(),
                next_seq: 0,
                pending: VecDeque::new(),
                pending_bytes: 0,
                kept: false,
            });
        }
        state.current.as_mut()
    }

    fn write_session(current: &Current) {
        let key = encode(&current.session.id);
        current
            .database
            .replace(SESSIONS, &key, &encode(&current.session), Origin::Local);
    }

    pub(super) fn encode<V: Serialize + ?Sized>(value: &V) -> Vec<u8> {
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

    pub(super) fn offset(_session: super::SessionId) -> Option<core::time::Duration> {
        None
    }

    pub(super) fn write<V: serde::Serialize>(
        _namespace: &str,
        _effect: super::RecordEffect,
        _stamped: &super::Stamped<V>,
    ) -> bool {
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

    /// A record of the kind the tests write.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct Test(u32);

    impl Record for Test {
        const KIND: &'static str = "test";
    }

    /// Another kind, sharing the session's sequence with [`Test`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct Other(u32);

    impl Record for Other {
        const KIND: &'static str = "other";
    }

    fn tests(database: &Database) -> Vec<Stamped<Test>> {
        Records::new(database).read()
    }

    fn sessions(database: &Database) -> Vec<Session> {
        Records::new(database).sessions()
    }

    #[test]
    #[serial]
    fn records_are_stamped_in_order_within_one_session() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let first = write(RecordEffect::Changed, &Test(1)).expect("recorded");
        let second = write(RecordEffect::Changed, &Other(2)).expect("recorded");

        assert_eq!(first.session, second.session);
        assert_eq!(second.seq, first.seq + 1);
        assert!(second.offset >= first.offset);
        assert_eq!(tests(&database)[0].record, Test(1));
        assert_eq!(Records::new(&database).read::<Other>()[0].stamp, second);
    }

    #[test]
    #[serial]
    fn a_session_is_written_once_and_carries_its_label() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        label("models build test");
        write(RecordEffect::Changed, &Test(1)).expect("recorded");

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
            write(RecordEffect::Changed, &Test(1)).expect("recorded")
        };

        let marks = Records::new(&database).read::<MarkRecord>();
        assert_eq!(marks.len(), 1);
        let span = &marks[0];
        assert_eq!(span.record.label, "load weights");
        assert!(span.stamp.seq < inner.seq);
        assert!(span.stamp.offset + span.record.wall >= inner.offset);
    }

    /// A session that only observes — a warm-up loading every kernel from
    /// the store — leaves nothing in the environment.
    #[test]
    #[serial]
    fn a_session_that_changes_nothing_leaves_nothing() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        {
            let _phase = mark("walk");
            write(RecordEffect::Observed, &Test(1)).expect("stamped");
        }

        assert!(sessions(&database).is_empty());
        assert!(tests(&database).is_empty());
        assert!(Records::new(&database).read::<MarkRecord>().is_empty());
    }

    /// The first change keeps the session, and what it observed before is
    /// written with it, in order.
    #[test]
    #[serial]
    fn a_change_keeps_what_the_session_observed_before_it() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        label("models build test");
        let observed = write(RecordEffect::Observed, &Test(1)).expect("stamped");
        let changed = write(RecordEffect::Changed, &Test(2)).expect("stamped");
        write(RecordEffect::Observed, &Test(3)).expect("stamped");

        let kept = tests(&database);
        let values: Vec<u32> = kept.iter().map(|record| record.record.0).collect();
        assert_eq!(values, vec![1, 2, 3]);
        assert_eq!(kept[0].stamp, observed);
        assert_eq!(kept[1].stamp, changed);
        let sessions = sessions(&database);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].label.as_deref(), Some("models build test"));
    }

    #[test]
    #[serial]
    fn nothing_is_recorded_when_off() {
        let (_dir, database) = fresh(RecordLevel::Off, None);
        assert_eq!(write(RecordEffect::Changed, &Test(1)), None);
        assert!(database.namespaces().is_empty());
    }

    #[test]
    #[serial]
    fn a_switch_starts_a_new_session_and_pruning_keeps_the_newest() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let first = write(RecordEffect::Changed, &Test(1)).expect("recorded");
        // Re-activating is a switch: the same database, a new session.
        crate::environment::activate(crate::environment::active());
        let second = write(RecordEffect::Changed, &Test(2)).expect("recorded");
        assert_ne!(first.session, second.session);
        assert_eq!(tests(&database).len(), 2);

        assert_eq!(Records::new(&database).prune(1), 1);
        let kept = tests(&database);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].stamp.session, second.session);
    }

    /// A process whose session another one pruned while it ran keeps writing
    /// under it: the next prune takes those records too.
    #[test]
    #[serial]
    fn a_prune_takes_the_records_of_a_session_pruned_while_it_ran() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let orphaned = write(RecordEffect::Changed, &Test(1)).expect("recorded");
        // Another process prunes this session's row while it runs on.
        database.purge_key(SESSIONS, &imp::encode(&orphaned.session));
        write(RecordEffect::Changed, &Test(2)).expect("recorded");

        crate::environment::activate(crate::environment::active());
        write(RecordEffect::Changed, &Test(3)).expect("recorded");
        crate::environment::activate(crate::environment::active());
        let newest = write(RecordEffect::Changed, &Test(4)).expect("recorded");

        assert_eq!(Records::new(&database).prune(1), 1);
        let kept = tests(&database);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].stamp, newest);
    }

    /// A record heavy enough to reach the budget in a few writes.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Heavy {
        index: u32,
        payload: Vec<u8>,
    }

    impl Record for Heavy {
        const KIND: &'static str = "heavy";
    }

    /// A session that changes nothing for a long while holds a bounded amount:
    /// the oldest of what it observed goes first, and a change keeps the rest.
    #[test]
    #[serial]
    fn a_session_holds_a_bounded_amount_before_it_changes_anything() {
        let (_dir, database) = fresh(RecordLevel::Basic, None);
        let observed = 64;
        for index in 0..observed {
            let heavy = Heavy {
                index,
                payload: vec![0; 64 << 10],
            };
            write(RecordEffect::Observed, &heavy).expect("stamped");
        }
        write(RecordEffect::Changed, &Test(0)).expect("stamped");

        let kept = Records::new(&database).read::<Heavy>();
        assert!(!kept.is_empty() && kept.len() < observed as usize);
        assert_eq!(kept.last().expect("kept").record.index, observed - 1);
        let indices: Vec<u32> = kept.iter().map(|heavy| heavy.record.index).collect();
        assert!(
            indices.windows(2).all(|pair| pair[1] == pair[0] + 1),
            "the newest, in order: {indices:?}"
        );
    }
}

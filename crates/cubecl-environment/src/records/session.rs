//! The session in progress, where there is a database to record into.

use super::{
    RecordEffect, RecordLevel, Records, RecordsConfig, SESSIONS, Session, SessionId, Stamp, Stamped,
};
use crate::bytes::Bytes;
use crate::persistence::{Database, Insertion, Origin, encode};
use crate::sync::{AtomicU8, LazyLock, Mutex, Ordering};
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use core::time::Duration;
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
    key: Bytes,
    value: Bytes,
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

pub(crate) fn configure(config: RecordsConfig) {
    LEVEL.store(config.level as u8, Ordering::Relaxed);
    STATE.lock().keep_sessions = config.keep_sessions;
}

pub(crate) fn level() -> RecordLevel {
    match LEVEL.load(Ordering::Relaxed) {
        0 => RecordLevel::Off,
        1 => RecordLevel::Basic,
        _ => RecordLevel::Full,
    }
}

pub(crate) fn label(label: String) {
    let mut state = STATE.lock();
    state.label = Some(label.clone());
    if let Some(current) = state.current.as_mut() {
        current.session.label = Some(label);
        if current.kept {
            write_session(current);
        }
    }
}

pub(crate) fn stamp() -> Option<Stamp> {
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
pub(crate) fn offset(session: SessionId) -> Option<Duration> {
    let state = STATE.lock();
    let current = state.current.as_ref()?;
    let live =
        current.session.id == session && current.generation == crate::environment::generation();
    live.then(|| current.started.elapsed())
}

pub(crate) fn write<V: Serialize>(
    namespace: &str,
    effect: RecordEffect,
    stamped: &Stamped<V>,
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
        Insertion::Stored
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
            session: start_session(state.label.clone()),
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

/// The session a process starts now, under `label`.
fn start_session(label: Option<String>) -> Session {
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Session {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::{Mark, MarkRecord, Record, configure, label, write};
    use alloc::vec;
    use alloc::vec::Vec;
    use serde::Deserialize;
    use serial_test::serial;

    /// Points the active environment at a fresh database, recording at
    /// `level`, and hands back the database to read.
    fn fresh(level: RecordLevel, keep: Option<u32>) -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().expect("a temp dir");
        crate::environment::set_root(dir.path());
        crate::environment::activate(alloc::format!("records-{}", std::process::id()));
        configure(RecordsConfig {
            level,
            keep_sessions: keep,
        });
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
            let _phase = Mark::new("load weights");
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
            let _phase = Mark::new("walk");
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
        database.purge_key(SESSIONS, &encode(&orphaned.session));
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

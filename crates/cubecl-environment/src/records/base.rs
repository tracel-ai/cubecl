#[cfg(not(native_cache))]
use super::disabled as session;
#[cfg(native_cache)]
use super::session;
use alloc::string::String;
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

/// How much an environment records of its own build, and how long it keeps
/// it: what [`configure`] takes, and the `[environment.records]` table of the
/// runtime's configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordsConfig {
    /// `basic` by default: every record, without the heavy parts.
    #[serde(default)]
    pub level: RecordLevel,

    /// How many sessions an environment keeps; the oldest beyond it are
    /// pruned with their records when a session is kept, at its first change
    /// to the environment. A session that changes nothing prunes nothing, and
    /// the session being kept always survives, so `0` keeps it alone, as `1`
    /// does. Every session is kept when unset.
    #[serde(default)]
    pub keep_sessions: Option<u32>,
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
    /// The kind records of this type are written under. Any but `sessions`,
    /// which names the [namespace sessions are written to](SESSIONS).
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

/// Sets how much is recorded, and how many sessions an environment keeps.
/// Called by the runtime once its configuration is loaded.
pub fn configure(config: RecordsConfig) {
    session::configure(config);
}

/// The level in force.
pub fn level() -> RecordLevel {
    session::level()
}

/// Whether anything is recorded.
pub fn enabled() -> bool {
    level() != RecordLevel::Off
}

/// Names what this process is doing, for the session in progress and every
/// later one: `models build qwen3-8b`, `serve`.
pub fn label<S: Into<String>>(label: S) {
    session::label(label.into());
}

/// Records `record` under its [namespace](Record::namespace) in the active
/// environment, stamped now: written if the session is kept, which an
/// [`Observed`](RecordEffect::Observed) record alone does not decide. `None`
/// when recording is off or there is no database to write to.
pub fn write<R: Record + Serialize>(effect: RecordEffect, record: &R) -> Option<Stamp> {
    let stamp = session::stamp()?;
    write_stamped(stamp, effect, record).then_some(stamp)
}

/// Records `record` with a stamp taken earlier. `false` when the session it
/// was stamped in is gone, or the write failed.
pub(crate) fn write_stamped<R: Record + Serialize>(
    stamp: Stamp,
    effect: RecordEffect,
    record: &R,
) -> bool {
    let namespace = R::namespace();
    // A record under the sessions' namespace would be read as a session, and
    // skipped by every prune.
    debug_assert_ne!(namespace, SESSIONS, "`sessions` is not a record kind");
    session::write(&namespace, effect, &Stamped { stamp, record })
}

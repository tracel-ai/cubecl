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

mod base;
mod span;

// The session in progress: where there is a database to record into, and the
// no-op standing in for it where there is none.
#[cfg(not(native_cache))]
mod disabled;
#[cfg(native_cache)]
mod reader;
#[cfg(native_cache)]
mod session;

pub use base::*;
#[cfg(native_cache)]
pub use reader::Records;
pub use span::{Mark, MarkRecord, Span};

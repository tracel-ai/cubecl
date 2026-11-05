//! The SQL engine under an environment's database: the few calls the
//! persistence makes of one, each engine answering them alike.
//!
//! Turso is the engine by default, and the only one in the browser. The
//! `rusqlite` feature puts `SQLite` under the native files instead. Features
//! add up across a build, so a build in which any crate asks for `rusqlite`
//! has `SQLite` serve every native file: Turso is still compiled, for the
//! browser, but opens nothing natively.

mod base;

pub(crate) use base::*;

#[cfg(turso_engine)]
mod turso;

#[cfg(turso_engine)]
pub(crate) use self::turso::*;

#[cfg(sqlite_engine)]
mod rusqlite;

#[cfg(sqlite_engine)]
pub(crate) use self::rusqlite::*;

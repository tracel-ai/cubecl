use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Deref;

use super::Connection;

/// A value bound to one parameter of a statement.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Param<'a> {
    Text(&'a str),
    Blob(&'a [u8]),
    Integer(i64),
}

impl<'a> From<&'a str> for Param<'a> {
    fn from(text: &'a str) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a String> for Param<'a> {
    fn from(text: &'a String) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a [u8]> for Param<'a> {
    fn from(blob: &'a [u8]) -> Self {
        Self::Blob(blob)
    }
}

impl<'a> From<&'a Vec<u8>> for Param<'a> {
    fn from(blob: &'a Vec<u8>) -> Self {
        Self::Blob(blob)
    }
}

impl From<i64> for Param<'_> {
    fn from(integer: i64) -> Self {
        Self::Integer(integer)
    }
}

/// Why the engine refused a call, in the engine's words, named for what the
/// persistence does about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SqlError {
    /// Another connection held what the call needed past the busy timeout.
    Busy(String),
    /// A file opened for this process alone is open in another process.
    HeldByAnotherProcess(String),
    /// The file is damaged.
    Corrupt(String),
    /// The file is not a database.
    NotADatabase(String),
    /// The file was opened read-only, or can't be written.
    ReadOnly(String),
    /// The disk is full.
    Full(String),
    /// The disk refused a read or a write.
    Io(String),
    /// Anything else.
    Other(String),
}

impl core::fmt::Display for SqlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Busy(message)
            | Self::HeldByAnotherProcess(message)
            | Self::Corrupt(message)
            | Self::NotADatabase(message)
            | Self::ReadOnly(message)
            | Self::Full(message)
            | Self::Io(message)
            | Self::Other(message) => f.write_str(message),
        }
    }
}

/// A failure as the persistence reports it: a message.
impl From<SqlError> for String {
    fn from(error: SqlError) -> Self {
        alloc::string::ToString::to_string(&error)
    }
}

/// How a database file is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    browser_cache,
    expect(dead_code, reason = "the browser opens its files one way")
)]
pub(crate) enum Opening {
    /// Read and never written.
    ReadOnly,
    /// Read and written, by this process and any other sharing the file.
    Shared,
    /// Read and written by this process alone, which the engine refuses
    /// while another process has the file open
    /// ([`SqlError::HeldByAnotherProcess`]).
    ProcessLocal,
    /// A file this process alone writes and may compact (`VACUUM`): a bundle
    /// being staged.
    Staging,
}

/// How a transaction takes the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Behavior {
    /// On its first write.
    #[cfg_attr(
        not(browser_cache),
        expect(dead_code, reason = "only the browser's writes defer")
    )]
    Deferred,
    /// Up front, so two writers wait on each other rather than fail halfway.
    #[cfg_attr(
        browser_cache,
        expect(dead_code, reason = "the browser's writes defer")
    )]
    Immediate,
}

impl Behavior {
    fn begin(self) -> &'static str {
        match self {
            Self::Deferred => "BEGIN DEFERRED",
            Self::Immediate => "BEGIN IMMEDIATE",
        }
    }
}

/// A transaction on a connection, rolled back when dropped before it is
/// committed.
pub(crate) struct Transaction<'c> {
    connection: &'c Connection,
    open: bool,
}

impl<'c> Transaction<'c> {
    /// Begins a transaction on `connection`.
    pub(crate) fn new(connection: &'c Connection, behavior: Behavior) -> Result<Self, SqlError> {
        connection.execute(behavior.begin(), &[])?;
        Ok(Self {
            connection,
            open: true,
        })
    }

    /// Commits the transaction; one that fails to commit is rolled back.
    pub(crate) fn commit(mut self) -> Result<(), SqlError> {
        self.connection.execute("COMMIT", &[])?;
        self.open = false;
        Ok(())
    }
}

impl Deref for Transaction<'_> {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.connection
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if self.open {
            let _ = self.connection.execute("ROLLBACK", &[]);
        }
    }
}

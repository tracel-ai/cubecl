//! Access policy and reader/writer configuration for [`Bytes`](super::Bytes).
//!
//! Accessing a [`Bytes`] is parameterized by an [`AccessPolicy`] that dictates what the
//! backing [`AllocationController`](super::AllocationController) is *allowed* to do to satisfy
//! the access. The policy makes the previously-implicit decisions explicit:
//!
//! - **copy-on-read**: materializing lazy storage (file / device) into a host buffer.
//! - **copy-on-write**: copying shared storage into a private buffer before mutation.
//!
//! A [`Reader`]/[`Writer`] is a small builder over a policy, used with
//! [`Bytes::read`](super::Bytes::read) / [`Bytes::write`](super::Bytes::write).

use alloc::string::String;

/// What an access is allowed to do to satisfy a read/write of a [`Bytes`](super::Bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessPolicy {
    allow_copy: bool,
}

impl Default for AccessPolicy {
    fn default() -> Self {
        Self::allow_copy()
    }
}

impl AccessPolicy {
    /// Allow allocating/copying a new buffer to satisfy the access (copy-on-read
    /// materialization and copy-on-write).
    pub fn allow_copy() -> Self {
        Self { allow_copy: true }
    }

    /// Forbid any copy: the access succeeds only if it can be served from already-resident
    /// memory, otherwise it fails with [`AccessError::WouldCopy`].
    pub fn zero_copy() -> Self {
        Self { allow_copy: false }
    }

    /// Whether copying is allowed to satisfy the access.
    pub fn copy_allowed(&self) -> bool {
        self.allow_copy
    }
}

/// Error returned when accessing a [`Bytes`](super::Bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessError {
    /// The access required allocating/copying a new buffer, which the [`AccessPolicy`]
    /// forbade.
    WouldCopy,
    /// Materializing the data failed (e.g. a file or device read error). The payload is a
    /// human-readable message.
    Read(String),
}

impl core::fmt::Display for AccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WouldCopy => f.write_str("access would require a copy, which the policy forbids"),
            Self::Read(msg) => write!(f, "failed to materialize bytes: {msg}"),
        }
    }
}

impl core::error::Error for AccessError {}

/// Configures a read of a [`Bytes`](super::Bytes::read).
///
/// Defaults to allowing copies (the same behavior as `Deref`). Use [`Reader::no_copy`] to
/// require resident memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reader {
    pub(crate) policy: AccessPolicy,
}

impl Reader {
    /// A reader that allows copies to satisfy the read.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require the read to be served without any copy/materialization.
    pub fn no_copy(mut self) -> Self {
        self.policy = AccessPolicy::zero_copy();
        self
    }
}

/// Configures a mutable access of a [`Bytes`](super::Bytes::write).
///
/// Defaults to allowing copies (copy-on-write). Use [`Writer::no_copy`] to require that the
/// buffer is already private/mutable.
#[derive(Clone, Copy, Debug, Default)]
pub struct Writer {
    pub(crate) policy: AccessPolicy,
}

impl Writer {
    /// A writer that allows copy-on-write to satisfy the access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require mutable access without any copy-on-write (fails on still-shared buffers).
    pub fn no_copy(mut self) -> Self {
        self.policy = AccessPolicy::zero_copy();
        self
    }
}

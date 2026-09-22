//! The device side of a relocation.

use super::StorageCopy;
use crate::server::{IoError, ServerError};

/// Where a relocation's bytes are copied, and what it waits on.
///
/// The one part of a relocation a backend supplies: everything else — what
/// moves, in which order, and when the move is committed — is the same
/// whichever device is underneath.
pub trait CopyQueue {
    /// Wait until the device is done with everything that could still touch
    /// the allocations about to move: every stream, and whatever the driver
    /// runs outside them.
    ///
    /// # Errors
    ///
    /// The fault a wait revealed.
    fn wait_device(&mut self) -> Result<(), ServerError>;

    /// Enqueue copying `copy.source`'s bytes into `copy.target`.
    ///
    /// # Errors
    ///
    /// The device's refusal to copy.
    fn copy(&mut self, copy: &StorageCopy) -> Result<(), IoError>;

    /// Wait until the copies enqueued so far have landed.
    ///
    /// # Errors
    ///
    /// The fault the wait revealed.
    fn wait_copies(&mut self) -> Result<(), ServerError>;
}

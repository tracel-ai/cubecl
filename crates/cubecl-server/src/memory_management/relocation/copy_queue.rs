//! The device side of a relocation.

use super::StorageCopy;
use crate::server::{IoError, ServerError};
use crate::storage::ComputeStorage;

/// Where a relocation's bytes are copied, and what it waits on.
///
/// The one part of a relocation a runtime supplies: everything else — when to
/// move, what moves, and when the move is committed — is the same whichever
/// device is underneath. It holds the device queue the copies go on, never the
/// memory the relocation is moving, so a reservation can hand it the storage
/// it is already holding.
pub trait CopyQueue<Storage: ComputeStorage> {
    /// Wait until the device is done with everything that could still touch
    /// the allocations about to move: every stream, and whatever the driver
    /// runs outside them.
    ///
    /// # Errors
    ///
    /// The fault a wait revealed.
    fn wait_device(&mut self) -> Result<(), ServerError>;

    /// Add copying `copy.source`'s bytes into `copy.target` to the batch
    /// [`wait_copies`](Self::wait_copies) lands. A queue may hold it until
    /// then, so one relocation is one submission however many copies it
    /// makes.
    ///
    /// # Errors
    ///
    /// The device's refusal to copy.
    fn copy(&mut self, storage: &mut Storage, copy: &StorageCopy) -> Result<(), IoError>;

    /// Submit the copies added so far, if the queue held them, and wait until
    /// they have landed.
    ///
    /// # Errors
    ///
    /// The fault the wait revealed.
    fn wait_copies(&mut self) -> Result<(), ServerError>;
}

//! The device side of a relocation.

use super::StorageCopy;
use crate::server::{IoError, ServerError};
use crate::storage::StorageAllocator;

/// Copies a relocation's bytes, and waits on the device around them.
///
/// The one part of a relocation a runtime supplies: everything else — when to
/// move, what moves, and when the move is committed — is the same whichever
/// device is underneath. It holds the device queue the copies go on, never the
/// memory the relocation is moving, so a reservation can hand it the storage
/// it is already holding.
///
/// # Safety
///
/// An implementation promises what [`wait_device`](Self::wait_device) and
/// [`wait_copies`](Self::wait_copies) guarantee, for the device it was built
/// from, establishing it itself or from a proof it is built with: the
/// copies a relocation enqueues rely on both, so a copier that answers `Ok`
/// early lets a copy race the work that reads or writes its memory. A host
/// copier meets both by having the work finished when it answers.
pub unsafe trait RelocationCopier<Storage: StorageAllocator> {
    /// Wait until the device is done with everything that could still touch
    /// the allocations about to move: every stream, and whatever the driver
    /// runs outside them.
    ///
    /// # Guarantees
    ///
    /// `Ok` only once every copy issued after it returns runs after all the
    /// work issued on the device before it.
    ///
    /// # Errors
    ///
    /// The fault a wait revealed.
    fn wait_device(&mut self) -> Result<(), ServerError>;

    /// Add copying `copy.source`'s bytes into `copy.target` to the batch
    /// [`wait_copies`](Self::wait_copies) lands. An implementation may hold
    /// it until then, so one relocation is one submission however many
    /// copies it makes.
    ///
    /// # Errors
    ///
    /// The device's refusal to copy.
    ///
    /// # Safety
    ///
    /// - `copy.source` and `copy.target` are storages of `storage`, of the
    ///   same size, on distinct live device allocations kept alive until
    ///   [`wait_copies`](Self::wait_copies) returns.
    /// - Each is followed, within its device allocation, by the padding that
    ///   rounds its size up to the device's alignment: a copier may copy that
    ///   rounded size, writing into the target's padding.
    unsafe fn copy(&mut self, storage: &mut Storage, copy: &StorageCopy) -> Result<(), IoError>;

    /// Submit the copies added so far, if they were held, and wait until they
    /// have landed.
    ///
    /// # Guarantees
    ///
    /// `Ok` only once every use of the storages issued after it returns runs
    /// after every copy added.
    ///
    /// # Errors
    ///
    /// The fault the wait revealed.
    fn wait_copies(&mut self) -> Result<(), ServerError>;
}

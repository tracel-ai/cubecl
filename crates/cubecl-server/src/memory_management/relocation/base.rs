//! Moving live allocations off outdated pages, so those pages go back to the
//! driver now rather than when their longest-lived allocation ends.
//!
//! A relocation is planned by the pools it empties, landed on the device by a
//! [`RelocationLander`] through a [`RelocationCopier`], and committed by those
//! pools again from the [`LandedRelocation`] the lander hands back: an
//! allocation cannot move before its bytes are where it moves to.

use super::{LandedRelocation, RelocationCopier};
use crate::memory_management::{
    ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation, release_or_leak,
};
use crate::server::ServerError;
use crate::storage::StorageAllocator;
use crate::storage::StorageHandle;
use alloc::vec::Vec;

/// Lands a relocation: copies the bytes of live allocations into the slices
/// reserved to receive them.
///
/// # Guarantees
///
/// Nothing has moved before [`land`](Self::land) returns: dropping it abandons
/// the relocation with nothing lost, the reserved slices freed with it.
#[derive(Debug)]
pub(crate) struct RelocationLander {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

/// Where a relocation may reserve the slices it moves allocations into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRoom {
    /// Only the room the current pages already have. Memory is what is short,
    /// and a relocation that allocated a page would spend more than it frees.
    Held,
    /// New pages of the current size too, when the room held runs out. Memory
    /// is not what is short, so emptying every outdated page is worth a page.
    MayAllocate,
}

/// One device copy a relocation needs, between two resolved storages of the
/// same size.
#[derive(Debug, Clone)]
pub struct StorageCopy {
    /// Where the bytes are now.
    pub source: StorageHandle,
    /// Where they go.
    pub target: StorageHandle,
}

/// One live allocation moving to a reserved slice: `allocation`'s handle is
/// handed over to the target once `copy` has landed.
#[derive(Debug)]
pub(crate) struct Move {
    /// The allocation being moved, bound rather than held, so the move does not count among
    /// its owners' handles while it runs.
    pub allocation: ManagedMemoryBinding,
    /// The slice reserved to receive it.
    pub target: ManagedMemoryHandle,
    /// The bytes to copy, or `None` when the source was carved under a process mode
    /// that discards launches and never resolved: there is nothing behind it to
    /// copy.
    pub copy: Option<StorageCopy>,
}

impl RelocationLander {
    /// A relocation of `moves`, each with its target already reserved.
    ///
    /// # Safety
    ///
    /// Every copy of `moves` is between the storage of a live allocation and
    /// the storage of the slice reserved for it: of the same size, on distinct
    /// device allocations of the storage later passed to [`land`](Self::land),
    /// each padded up to the device's alignment, the target held by nobody
    /// but its move. Only the pools of memory management plan moves, so the
    /// contract never leaves it.
    pub(in crate::memory_management) unsafe fn new(moves: Vec<Move>) -> Self {
        Self { moves }
    }

    /// Whether there is nothing to move.
    pub(crate) fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Copy every allocation's bytes with `copier`, waiting for the device
    /// before the first copy and for the copies after the last.
    ///
    /// # Guarantees
    ///
    /// With nothing to move, waits on nothing.
    ///
    /// The wait for the copies happens whether every one of them was enqueued
    /// or not, so a relocation abandoned here frees a target only once every
    /// later use of it runs after the copies: when that wait fails on a
    /// device that is not poisoned, the targets and the allocations they were
    /// for stay reserved for good.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault a wait revealed. The
    /// relocation is abandoned either way.
    pub(crate) fn land<Storage: StorageAllocator>(
        self,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
    ) -> Result<LandedRelocation, ServerError> {
        if self.moves.is_empty() {
            return Ok(LandedRelocation::new(self.moves));
        }
        copier.wait_device()?;
        let enqueued = self
            .moves
            .iter()
            .filter_map(|relocated| relocated.copy.as_ref())
            // SAFETY: `new`'s contract makes each copy one between storages of
            // `storage`, of the same size on distinct allocations, padded up to
            // the device's alignment, its target held by nobody but its move.
            // Both stay alive until `wait_copies` returns: the binding keeps
            // nothing alive, but the source's page can be neither released nor
            // carved again while this runs, since the caller holds its pool
            // mutably borrowed for the whole call; the target handle keeps the
            // target slice taken. No other work is issued while the copier and
            // the storage are borrowed here, and the copier orders the copies
            // after every earlier work and before every later one.
            .try_for_each(|copy| unsafe { copier.copy(storage, copy) });
        if let Err(err) = copier.wait_copies() {
            release_or_leak(self.moves, &err);
            enqueued?;
            return Err(err);
        }
        enqueued?;
        Ok(LandedRelocation::new(self.moves))
    }
}

impl Move {
    /// The location the allocation moves from and the one it moves to, read
    /// from the handles, which the pools keep current: a move stores no
    /// location of its own, so it can never hold a stale one.
    pub(crate) fn locations(&self) -> (MemoryLocation, MemoryLocation) {
        (
            self.allocation.descriptor().location(),
            self.target.descriptor().location(),
        )
    }
}

#[cfg(all(test, feature = "storage-bytes"))]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;
    use crate::{driver::DriverError, server::IoError, storage::BytesStorage};

    /// Copies the device refuses, or a wait for them that fails.
    struct FailingRelocationCopier {
        device: Result<(), ()>,
        copy: Result<(), ()>,
        wait: Result<(), ()>,
    }

    // SAFETY: test memory that no work reads or writes, and copies that do
    // nothing or land on return.
    unsafe impl RelocationCopier<BytesStorage> for FailingRelocationCopier {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            self.device
                .map_err(|()| DriverError::new("wait device", 201).into())
        }

        unsafe fn copy(
            &mut self,
            storage: &mut BytesStorage,
            copy: &StorageCopy,
        ) -> Result<(), IoError> {
            self.copy.map_err(|()| IoError::Unknown {
                description: "refused".into(),
                backtrace: Default::default(),
            })?;
            let source = storage.get(&copy.source);
            let mut target = storage.get(&copy.target);
            target.write().copy_from_slice(source.read());
            Ok(())
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            self.wait.map_err(|()| DriverError::new("wait", 201).into())
        }
    }

    /// A relocation of one allocation, its source and target handles.
    fn relocation(
        storage: &mut BytesStorage,
    ) -> (RelocationLander, ManagedMemoryHandle, ManagedMemoryHandle) {
        let source = ManagedMemoryHandle::new();
        let target = ManagedMemoryHandle::new();
        let relocation = alloc::vec![Move {
            allocation: source.to_binding(),
            target: target.clone(),
            copy: Some(StorageCopy {
                source: storage
                    .allocate_bytes(4)
                    .expect("the host should have 4 bytes"),
                target: storage
                    .allocate_bytes(4)
                    .expect("the host should have 4 bytes"),
            }),
        }];
        // SAFETY: two distinct 4-byte host allocations no work touches, held
        // by the moves until the relocation lands.
        (unsafe { RelocationLander::new(relocation) }, source, target)
    }

    #[test]
    fn a_failed_wait_keeps_the_moves_reserved() {
        let mut storage = BytesStorage::default();
        let (relocation, source, target) = relocation(&mut storage);
        let mut copier = FailingRelocationCopier {
            device: Ok(()),
            copy: Ok(()),
            wait: Err(()),
        };

        assert!(relocation.land(&mut storage, &mut copier).is_err());
        assert!(!source.is_free());
        assert!(!target.is_free());
    }

    /// A refused copy abandons the relocation once the wait for the copies
    /// lands, freeing its targets.
    #[test]
    fn a_refused_copy_abandons_the_relocation() {
        let mut storage = BytesStorage::default();
        let (relocation, _source, target) = relocation(&mut storage);
        let mut copier = FailingRelocationCopier {
            device: Ok(()),
            copy: Err(()),
            wait: Ok(()),
        };

        let refused = relocation.land(&mut storage, &mut copier);

        assert!(matches!(refused, Err(ServerError::Io(_))), "{refused:?}");
        assert!(target.is_free(), "the target is free once abandoned");
    }

    /// Copies that land hand every move back.
    #[test]
    fn landed_copies_hand_every_move_back() {
        let mut storage = BytesStorage::default();
        let (relocation, _source, _target) = relocation(&mut storage);
        let mut copier = FailingRelocationCopier {
            device: Ok(()),
            copy: Ok(()),
            wait: Ok(()),
        };

        let landed = relocation
            .land(&mut storage, &mut copier)
            .expect("the copies should land");

        assert_eq!(landed.into_moves().len(), 1);
    }

    /// A device that cannot be waited on copies nothing.
    #[test]
    fn a_failed_device_wait_copies_nothing() {
        let mut storage = BytesStorage::default();
        let (relocation, _source, target) = relocation(&mut storage);
        let mut copier = FailingRelocationCopier {
            device: Err(()),
            copy: Err(()),
            wait: Err(()),
        };

        let refused = relocation.land(&mut storage, &mut copier);

        assert!(refused.is_err());
        assert!(
            target.is_free(),
            "the target is freed with the abandoned relocation"
        );
    }

    /// A relocation with nothing to move waits on nothing.
    #[test]
    fn landing_nothing_waits_on_nothing() {
        let mut storage = BytesStorage::default();
        // SAFETY: no moves, so no copy to make.
        let relocation = unsafe { RelocationLander::new(Vec::new()) };
        let mut copier = FailingRelocationCopier {
            device: Err(()),
            copy: Err(()),
            wait: Err(()),
        };

        assert!(relocation.land(&mut storage, &mut copier).is_ok());
    }
}

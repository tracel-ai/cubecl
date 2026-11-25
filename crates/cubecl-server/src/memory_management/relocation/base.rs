//! Moving live allocations off outdated pages, so those pages go back to the
//! driver now rather than when their longest-lived allocation ends.
//!
//! A [`Relocation`] is planned by the pools it empties, copied on the device
//! through a [`RelocationCopier`], and committed by those pools again from the
//! [`LandedRelocation`] the copy step hands back: an allocation cannot move
//! before its bytes are where it moves to.

use super::{LandedRelocation, RelocationCopier};
use crate::memory_management::{
    ManagedMemoryBinding, ManagedMemoryHandle, MemoryLocation, release_or_leak,
};
use crate::server::ServerError;
use crate::storage::StorageAllocator;
use crate::storage::StorageHandle;
use alloc::vec::Vec;

/// Lands a relocation: copies the bytes of live allocations into the slices
/// reserved to receive them, and hands the moves back once they landed.
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
    pub(crate) fn new(moves: Vec<Move>) -> Self {
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
    /// The wait for the copies happens whether every one of them was enqueued
    /// or not, so a relocation abandoned here never frees a target a copy is
    /// still writing: when that wait fails on a device that is not poisoned,
    /// the targets and the allocations they were for stay reserved for good.
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
        copier.wait_device()?;
        let enqueued = self
            .moves
            .iter()
            .filter_map(|relocated| relocated.copy.as_ref())
            // SAFETY: each copy is between a live allocation and the slice
            // reserved for it, of its size, on another allocation: a target
            // is a fresh reservation on a current or kept-apart page, never
            // the page its source sits on. The device waited above, and the
            // copier is held here until `wait_copies` returns, so nothing
            // reads a target or writes a source before then.
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
    /// location of its own, so it can never hold a stale one. The handles are
    /// dropped before the locations are answered, so only the slices hold
    /// them once the pools hand the allocation over.
    pub(crate) fn into_locations(self) -> (MemoryLocation, MemoryLocation) {
        let source = self.allocation.descriptor().location();
        let destination = self.target.descriptor().location();
        (source, destination)
    }
}

#[cfg(all(test, feature = "storage-bytes"))]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;
    use crate::{driver::DriverError, server::IoError, storage::BytesStorage};

    /// Copies the device refuses, or a wait for them that fails.
    struct FailingCopier {
        copy: Result<(), ()>,
        wait: Result<(), ()>,
    }

    impl RelocationCopier<BytesStorage> for FailingCopier {
        fn wait_device(&mut self) -> Result<(), ServerError> {
            Ok(())
        }

        unsafe fn copy(&mut self, _: &mut BytesStorage, _: &StorageCopy) -> Result<(), IoError> {
            self.copy.map_err(|()| IoError::Unknown {
                description: "refused".into(),
                backtrace: Default::default(),
            })
        }

        fn wait_copies(&mut self) -> Result<(), ServerError> {
            self.wait.map_err(|()| DriverError::new("wait", 201).into())
        }
    }

    /// The moves of a relocation of one allocation, its source and target handles.
    fn relocation(
        storage: &mut BytesStorage,
    ) -> (Vec<Move>, ManagedMemoryHandle, ManagedMemoryHandle) {
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
        (relocation, source, target)
    }

    #[test]
    fn a_failed_wait_keeps_the_moves_reserved() {
        let mut storage = BytesStorage::default();
        let (relocation, source, target) = relocation(&mut storage);
        let mut copier = FailingCopier {
            copy: Ok(()),
            wait: Err(()),
        };

        assert!(
            RelocationLander::new(relocation)
                .land(&mut storage, &mut copier)
                .is_err()
        );
        assert!(!source.is_free());
        assert!(!target.is_free());
    }

    /// A refused copy abandons the relocation once the wait for the copies
    /// lands, freeing its targets.
    #[test]
    fn a_refused_copy_abandons_the_relocation() {
        let mut storage = BytesStorage::default();
        let (relocation, _source, target) = relocation(&mut storage);
        let mut copier = FailingCopier {
            copy: Err(()),
            wait: Ok(()),
        };

        let refused = RelocationLander::new(relocation).land(&mut storage, &mut copier);

        assert!(matches!(refused, Err(ServerError::Io(_))), "{refused:?}");
        assert!(target.is_free(), "the target is free once abandoned");
    }

    /// Copies that land hand every move back.
    #[test]
    fn landed_copies_hand_every_move_back() {
        let mut storage = BytesStorage::default();
        let (relocation, _source, _target) = relocation(&mut storage);
        let mut copier = FailingCopier {
            copy: Ok(()),
            wait: Ok(()),
        };

        let landed = RelocationLander::new(relocation)
            .land(&mut storage, &mut copier)
            .expect("the copies should land");

        assert_eq!(landed.into_moves().len(), 1);
    }
}

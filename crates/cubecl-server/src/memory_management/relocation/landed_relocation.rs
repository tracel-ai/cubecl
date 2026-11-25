use super::{Move, RelocationCopier};
use crate::memory_management::{MemoryLocation, release_or_leak};
use crate::server::ServerError;
use crate::storage::StorageAllocator;
use alloc::vec::Vec;

/// A [`Relocation`](super::Relocation) whose copies have landed: what the
/// pools commit, and only [`Relocation::copy`](super::Relocation::copy) makes.
#[derive(Debug)]
pub(crate) struct LandedRelocation {
    /// The moves, in the order they were planned (Arrival).
    moves: Vec<Move>,
}

impl LandedRelocation {
    /// Copy the bytes of every one of `moves` with `copier`, waiting for the
    /// device before the first copy and for the copies after the last, and
    /// hand the moves back landed once every copy did. The wait for the
    /// copies happens whether every one of them was enqueued or not, and when
    /// it fails on a device that is not poisoned, the moves are leaked rather
    /// than freed under a copy still writing.
    ///
    /// # Errors
    ///
    /// The first copy the device refused, or the fault a wait revealed.
    pub(super) fn land<Storage: StorageAllocator>(
        moves: Vec<Move>,
        storage: &mut Storage,
        copier: &mut dyn RelocationCopier<Storage>,
    ) -> Result<Self, ServerError> {
        copier.wait_device()?;
        let enqueued = moves
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
            release_or_leak(moves, &err);
            enqueued?;
            return Err(err);
        }
        enqueued?;
        Ok(Self { moves })
    }
}

/// The location each allocation of `landed` moves from and the one it moves
/// to, read from the handles, which the pools keep current: a move stores no
/// location of its own, so it can never hold a stale one. Each move's handles
/// are dropped before its locations are yielded, so only the slices hold them
/// once the pools hand the allocation over.
pub(crate) fn landed_locations(
    landed: LandedRelocation,
) -> impl Iterator<Item = (MemoryLocation, MemoryLocation)> {
    landed.moves.into_iter().map(|relocated| {
        let source = relocated.allocation.descriptor().location();
        let destination = relocated.target.descriptor().location();
        (source, destination)
    })
}

#[cfg(all(test, feature = "storage-bytes"))]
mod tests {
    use super::super::StorageCopy;
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

        assert!(LandedRelocation::land(relocation, &mut storage, &mut copier).is_err());
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

        let refused = LandedRelocation::land(relocation, &mut storage, &mut copier);

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

        let landed = LandedRelocation::land(relocation, &mut storage, &mut copier)
            .expect("the copies should land");

        assert_eq!(landed_locations(landed).count(), 1);
    }
}

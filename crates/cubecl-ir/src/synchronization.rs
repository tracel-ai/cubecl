use std::fmt::Display;

use type_hash::TypeHash;

/// All synchronization types.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Synchronization {
    // Synchronizize units in a cube.
    SyncUnits,
    SyncStorage,
}

impl Display for Synchronization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Synchronization::SyncUnits => write!(f, "sync_units()"),
            Synchronization::SyncStorage => write!(f, "sync_storage()"),
        }
    }
}

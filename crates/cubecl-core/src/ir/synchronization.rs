use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// All synchronization types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

use core::fmt::Display;

use crate::{OperationReflect, TypeHash};

/// All synchronization types.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = SyncOpCode)]
#[allow(missing_docs)]
pub enum Synchronization {
    // Synchronizize units in a cube.
    SyncCube,
    // Synchronize units within their plane
    SyncPlane,
    SyncStorage,
    /// Sync CTA proxy.
    /// Experimental, CUDA only, SM 9.0+ only
    SyncProxyShared,
}

impl Display for Synchronization {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Synchronization::SyncCube => write!(f, "sync_cube()"),
            Synchronization::SyncStorage => write!(f, "sync_storage()"),
            Synchronization::SyncProxyShared => write!(f, "sync_proxy_shared()"),
            Synchronization::SyncPlane => write!(f, "sync_plane()"),
        }
    }
}

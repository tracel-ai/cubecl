use crate::ir::Synchronization;

pub fn sync_units() {}

pub mod sync_units {
    use super::*;

    pub fn expand() -> Synchronization {
        Synchronization::SyncUnits
    }
}

pub fn sync_storage() {}

pub mod sync_storage {
    use super::*;

    pub fn expand() -> Synchronization {
        Synchronization::SyncStorage
    }
}

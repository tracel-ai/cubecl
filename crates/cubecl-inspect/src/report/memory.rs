use cubecl_environment::records::Stamped;
use cubecl_server::memory_management::MemoryRecord;
use serde::Serialize;

/// Every memory snapshot the file's sessions recorded, in the order they were
/// taken.
#[derive(Clone, Debug, Serialize)]
pub struct MemorySnapshots {
    pub snapshots: Vec<Stamped<MemoryRecord>>,
}

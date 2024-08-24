use crate::frontend::CubeContext;
use crate::ir::Synchronization;
// Among all backends, the memory order guarantee of WebGPU is the weakest
// So Cubecl's memory order cannot be stronger than that of WebGPU

/// workgroupBarrier()
/// See https://github.com/gpuweb/gpuweb/discussions/3935
pub fn sync_units() {}

pub mod sync_units {
    use super::*;

    pub fn __expand(context: &mut CubeContext) {
        context.register(Synchronization::SyncUnits)
    }
}

/// storageBarrier()
/// See https://github.com/gpuweb/gpuweb/discussions/3935
/// https://github.com/gpuweb/gpuweb/discussions/4821#discussioncomment-10397124
pub fn sync_storage() {}

pub mod sync_storage {
    use super::*;

    pub fn __expand(context: &mut CubeContext) {
        context.register(Synchronization::SyncStorage)
    }
}

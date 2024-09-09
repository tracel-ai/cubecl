use crate::frontend::CubeContext;
use crate::ir::Synchronization;
// Among all backends, the memory order guarantee of WebGPU is the weakest
// So Cubecl's memory order cannot be stronger than that of WebGPU

/// # Coordinates the following among all invocations in the current cube:
///
/// * Memory writes to variables in cube address space(shared memory) complete,
///   e.g. writes that were initiated actually land in the cube address space memory.
///
/// * Then all the invocations in the cube wait for each other to arrive at the barrier, i.e. this step.
///
/// * Then all the invocations int he cube begin executing after the barrier, and any writes to cube address space that were made before the barrier are now visible to any invocation in this cube.
pub fn sync_units() {}

pub mod sync_units {
    use super::*;

    pub fn expand(context: &mut CubeContext) {
        context.register(Synchronization::SyncUnits)
    }
}

/// * Sync_storage is the same but change "cube address space(shared memory)" to "storage address space(input args)". But the set of invocations that are collaborating is still only the invocations in the same cube.
///
/// * There is no guarantee about using barriers alone to make the writes to storage buffer in one cube become visible to invocations in a different cube.
pub fn sync_storage() {}

pub mod sync_storage {
    use super::*;

    pub fn expand(context: &mut CubeContext) {
        context.register(Synchronization::SyncStorage)
    }
}

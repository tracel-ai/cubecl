pub mod atomic;
pub mod bitwise;
pub mod branch;
pub mod cmp;
pub mod general;
pub mod math;
pub mod memory;
pub mod plane;
pub mod sync;
pub mod vector;

#[cfg(target_family = "wasm")]
pub(crate) fn ensure_linked() {
    // Pliron uses inventory for Wasm interface registration. A concrete reference
    // keeps each module's constructors in the final binary.
    atomic::ensure_linked();
    bitwise::ensure_linked();
    branch::ensure_linked();
    cmp::ensure_linked();
    general::ensure_linked();
    math::ensure_linked();
    memory::ensure_linked();
    plane::ensure_linked();
    sync::ensure_linked();
    vector::ensure_linked();
}

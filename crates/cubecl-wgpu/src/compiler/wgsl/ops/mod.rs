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
pub(crate) fn wasm_inventory_root() {
    atomic::wasm_inventory_root();
    bitwise::wasm_inventory_root();
    branch::wasm_inventory_root();
    cmp::wasm_inventory_root();
    general::wasm_inventory_root();
    math::wasm_inventory_root();
    memory::wasm_inventory_root();
    plane::wasm_inventory_root();
    sync::wasm_inventory_root();
    vector::wasm_inventory_root();
}

//! A compound assignment can't change a comptime binding with a runtime value: the binding is
//! plain Rust, so the update would be lost. Plain `=` is rejected the same way.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn compound_assign_runtime_into_comptime(
    perm: &[u32],
    output: &mut [u32],
    #[comptime] cycle: u32,
    #[comptime] has_perm: bool,
) {
    let mut word = cycle;
    if comptime!(has_perm) {
        word ^= perm[0];
    }
    word += perm[1];
    output[0] = word;
}

fn main() {}

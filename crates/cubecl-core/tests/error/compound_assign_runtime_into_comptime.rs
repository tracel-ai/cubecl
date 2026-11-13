//! A compound assignment can't change a comptime binding, with a runtime value or a comptime one:
//! the binding is plain Rust, so inside a runtime loop or branch the update would happen once
//! while the kernel expands, and the kernel would silently compute something else.
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
    let mut offset = cycle;
    for _ in 0..perm[2] {
        offset += cycle;
    }
    output[0] = word + offset;
}

fn main() {}

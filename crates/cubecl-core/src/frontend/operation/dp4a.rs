use crate::{prelude::*, unexpanded};

/// Packed int8×4 dot-product accumulate: `c + Σ a_i * b_i` for four signed bytes packed in `a`/`b`.
///
/// On CUDA this lowers to `__dp4a`. On Vulkan/SPIR-V it lowers to `OpSDot` plus add.
/// Other backends use a portable signed-byte polyfill.
#[allow(unused_variables)]
pub fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    unexpanded!()
}

/// Expand method of [`dp4a()`].
pub mod dp4a {
    use super::*;
    use cubecl_ir::{Arithmetic, FmaOperands, Instruction, Scope};

    pub fn expand(
        scope: &Scope,
        a: NativeExpand<i32>,
        b: NativeExpand<i32>,
        c: NativeExpand<i32>,
    ) -> NativeExpand<i32> {
        let output = scope.create_value(a.expand.value_type());
        let a = a.expand;
        let b = b.expand;
        let c = c.expand;

        scope.register(Instruction::new(
            Arithmetic::Dp4a(FmaOperands { a, b, c }),
            output,
        ));

        output.into()
    }
}

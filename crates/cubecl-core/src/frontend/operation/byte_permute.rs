use crate::{prelude::*, unexpanded};

/// Four bytes picked out of the eight of `low` and `high`, in each `u32` of a vector: byte `i` of
/// the result is the one that byte `i` of `selectors` names, `0..4` those of `low` and `4..8` those
/// of `high`, the least significant first. Only the low three bits of a selector are read.
///
/// The device's own instruction on AMD (`v_perm_b32`) and CUDA (`prmt`), and shifts elsewhere.
#[allow(unused_variables)]
pub fn byte_permute<C: CubePrimitive<Scalar = u32>>(low: C, high: C, selectors: C) -> C {
    unexpanded!()
}

/// Expand method of [`byte_permute()`].
pub mod byte_permute {
    use super::*;
    use crate::frontend::operation::base::binary_expand;
    use cubecl_ir::{
        Scope,
        dialect::bitwise::{BitwiseAndOp, BytePermuteOp},
    };

    /// The low three bits of every selector.
    const SELECTOR_BITS: u32 = 0x0707_0707;

    pub fn expand<C: CubePrimitive<Scalar = u32>>(
        scope: &Scope,
        low: NativeExpand<C>,
        high: NativeExpand<C>,
        selectors: NativeExpand<C>,
    ) -> NativeExpand<C> {
        let bits = NativeExpand::<u32>::from_lit(scope, SELECTOR_BITS);
        let selectors = binary_expand(scope, selectors.into(), bits.into(), BitwiseAndOp::new);

        let [low, high, selectors] = normalize_same_vectorization(
            scope,
            [
                low.read_value(scope),
                high.read_value(scope),
                selectors.read_value(scope),
            ],
        );
        let op = BytePermuteOp::new(scope.ctx_mut(), low, high, selectors);
        scope.register_with_result(&op).into()
    }
}

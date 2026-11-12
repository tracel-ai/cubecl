use crate::{prelude::*, unexpanded};

/// `c + a.x · b.x + a.y · b.y` for `a` and `b` two-wide vectors of `f16` or of `bf16`: each
/// product exact in `f32`, the sum in `f32`. One instruction where the device has a 16-bit pair
/// dot product accumulating in `f32`, two `f32` multiply-adds elsewhere.
#[allow(unused_variables)]
pub fn fdot2<V: CubePrimitive>(a: V, b: V, c: f32) -> f32 {
    unexpanded!()
}

/// Expand method of [`fdot2()`].
pub mod fdot2 {
    use super::*;
    use cubecl_ir::{Scope, dialect::math::Fdot2Op};

    pub fn expand<V: CubePrimitive>(
        scope: &Scope,
        a: NativeExpand<V>,
        b: NativeExpand<V>,
        c: NativeExpand<f32>,
    ) -> NativeExpand<f32> {
        let a = a.read_value(scope);
        let b = b.read_value(scope);
        let c = c.read_value(scope);
        let op = Fdot2Op::new(scope.ctx_mut(), a, b, c);
        scope.register_with_result(&op).into()
    }
}

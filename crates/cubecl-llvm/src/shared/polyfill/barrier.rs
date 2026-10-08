//! Target-specific lowering of the barrier copies.

use crate::prelude::*;
use cubecl_core::{
    ir::{
        dialect::{barrier::MemCopyAsyncOp, general::ReinterpretCastOp, memory::IndexOp},
        types::RuntimeArrayType,
    },
    prelude::*,
};

/// A cooperative `memcpy_async` is made by every unit of the cube, together. NVPTX lowers it as
/// each unit copying its own contiguous share, so it becomes a plain copy of that share here,
/// where the cube's position and size are still builtins the compiler folds.
#[op_interface_impl]
impl LowerOp for MemCopyAsyncOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        let nvptx = match ctx.target() {
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => true,
            _ => false,
        };
        nvptx && self.cooperative(ctx).0
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let (barrier, source, destination, length) = (
            self.barrier(ctx),
            self.source(ctx),
            self.destination(ctx),
            self.source_length(ctx),
        );
        let (start, count) = unit_share::expand(scope, length.into());
        let start = start.read_value(scope);
        let source = offset(scope, source, start);
        let destination = offset(scope, destination, start);
        let count = count.read_value(scope);
        let copy = MemCopyAsyncOp::new(scope.ctx_mut(), barrier, source, destination, count, false);
        scope.register(&copy);
        vec![]
    }
}

/// The `(start, count)` of the elements this unit copies of `length`: `ceil(length / units)` to
/// each unit, the last shares clamped to the end.
#[cube]
fn unit_share(length: usize) -> (usize, usize) {
    let units = CUBE_DIM as usize;
    let share = length.div_ceil(units);
    let start = UNIT_POS as usize * share;
    let start = select(start < length, start, length);
    let end = start + share;
    let end = select(end < length, end, length);
    (start, end - start)
}

/// `ptr`, a pointer to one element, moved `index` elements on.
fn offset(scope: &Scope, ptr: Value, index: Value) -> Value {
    let ctx = scope.ctx();
    let (elem, space) = {
        let ty = ptr.get_type(ctx).deref(ctx);
        let ptr_ty = ty
            .downcast_ref::<CubePointerType>()
            .expect("a copy operand is a pointer");
        (ptr_ty.inner, ptr_ty.address_space)
    };
    let array = RuntimeArrayType::get(ctx, elem).into();
    let array_ptr = CubePointerType::get(ctx, array, space).into();
    let as_array = ReinterpretCastOp::new(scope.ctx_mut(), array_ptr, ptr);
    let as_array = scope.register_with_result(&as_array);
    let element = IndexOp::new(scope.ctx_mut(), as_array, index, None);
    scope.register_with_result(&element)
}

use cubecl_ir::{Scope, dialect::general::BufferLenOp, pliron::value::Value};

use crate::frontend::buffer_idx;

pub fn expand_buffer_length_native(scope: &Scope, list: Value) -> Value {
    let buffer_idx = buffer_idx(scope, list);
    let buffer_length = BufferLenOp::new(scope.ctx_mut(), buffer_idx);
    scope.register_with_result(&buffer_length)
}

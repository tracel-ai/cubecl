use cubecl_ir::{
    Scope,
    dialect::general::BufferLenOp,
    pliron::{builtin::op_interfaces::OneResultInterface, value::Value},
};

use crate::frontend::buffer_idx;

pub fn expand_buffer_length_native(scope: &Scope, list: Value) -> Value {
    let buffer_idx = buffer_idx(scope, list);
    let buffer_length = BufferLenOp::new(scope.ctx_mut(), buffer_idx);
    scope.register(&buffer_length);
    buffer_length.get_result(scope.ctx())
}

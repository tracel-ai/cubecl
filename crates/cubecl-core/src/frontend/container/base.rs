use cubecl_ir::{
    Scope,
    dialect::general::BufferLenOp,
    pliron::{builtin::op_interfaces::OneResultInterface, value::Value},
};

pub fn expand_buffer_length_native(scope: &Scope, list: Value) -> Value {
    let buffer_length = BufferLenOp::new(&mut scope.ctx_mut(), list);
    scope.register(&buffer_length);
    buffer_length.get_result(&scope.ctx())
}

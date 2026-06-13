use crate::prelude::CubePrimitive;
use cubecl_ir::{Instruction, Metadata, Scope, Value};

pub fn expand_buffer_length_native(scope: &Scope, list: Value) -> Value {
    let out = scope.create_value(usize::__expand_as_type(scope));
    scope.register(Instruction::new(Metadata::BufferLength { list }, out));
    out
}

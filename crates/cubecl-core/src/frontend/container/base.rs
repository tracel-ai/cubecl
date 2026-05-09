use crate::prelude::CubePrimitive;
use cubecl_ir::{Instruction, Metadata, Scope, Variable};

pub fn expand_buffer_length_native(scope: &Scope, list: Variable) -> Variable {
    let out = scope.create_local(usize::__expand_as_type(scope));
    scope.register(Instruction::new(Metadata::BufferLength { var: list }, out));
    out
}

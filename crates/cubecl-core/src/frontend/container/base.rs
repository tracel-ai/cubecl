use crate::prelude::CubePrimitive;
use cubecl_ir::{Instruction, Metadata, Scope, Variable};

pub fn expand_length_native(scope: &mut Scope, list: Variable) -> Variable {
    let out = scope.create_local(usize::as_type(scope).into());
    scope.register(Instruction::new(
        Metadata::Length { var: list },
        out.clone().into(),
    ));
    out.into()
}

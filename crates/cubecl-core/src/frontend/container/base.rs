use cubecl_ir::{Elem, Instruction, Item, Metadata, Scope, UIntKind, Variable};

pub fn expand_length_native(scope: &mut Scope, list: Variable) -> Variable {
    let out = scope.create_local(Item::new(Elem::UInt(UIntKind::U32)));
    scope.register(Instruction::new(
        Metadata::Length { var: list },
        out.clone().into(),
    ));
    out.into()
}

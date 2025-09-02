use cubecl_ir::{ElemType, Instruction, Metadata, Scope, Type, UIntKind, Variable};

pub fn expand_length_native(scope: &mut Scope, list: Variable) -> Variable {
    let out = scope.create_local(Type::scalar(ElemType::UInt(UIntKind::U32)));
    scope.register(Instruction::new(
        Metadata::Length { var: list },
        out.clone().into(),
    ));
    out.into()
}

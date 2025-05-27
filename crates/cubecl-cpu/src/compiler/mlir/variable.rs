use cubecl_core::ir::{Variable, VariableKind};
use melior::{
    dialect::arith,
    ir::{BlockLike, Type, Value, attribute::IntegerAttribute},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: Variable, value: Value<'a, 'a>) {
        match variable.kind {
            VariableKind::LocalConst { id } => {
                self.current_local_variables.insert(id, value);
            }
            _ => todo!(),
        };
    }
    pub fn get_variable(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.global_buffers[id as usize]
            }
            VariableKind::LocalConst { id } => self
                .current_local_variables
                .get(&id)
                .expect("Variable should have been declared before")
                .clone(),
            VariableKind::Builtin(_builtin) => self.zero_index(),
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
    pub fn zero_index(&self) -> Value<'a, 'a> {
        let attribute = IntegerAttribute::new(Type::index(self.context), 0).into();
        self.block()
            .append_operation(arith::constant(self.context, attribute, self.location))
            .result(0)
            .unwrap()
            .into()
    }
}

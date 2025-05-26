use cubecl_core::ir::{Variable, VariableKind};
use melior::{
    dialect::{arith, memref, ods::vector},
    ir::{BlockLike, Type, Value, attribute::IntegerAttribute},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_variable(&mut self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            // VariableKind::GlobalInputArray(id) => {}
            VariableKind::GlobalOutputArray(id) => self.global_buffers[id as usize],
            VariableKind::LocalConst { id } => {
                let ptr_type = self.item_to_memref_type(variable.item);
                let operation =
                    memref::alloca(self.context, ptr_type, &[], &[], None, self.location);
                let value = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                self.current_local_variables.insert(id, value);
                value
            }
            // VariableKind::Builtin(builtin) => {}
            _ => todo!(),
        }
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
    pub fn get_variable_vector(&self, variable: Variable) -> Value<'a, 'a> {
        let value = self.get_variable(variable);
        let vector_type = self.item_to_type(variable.item);
        let operation = vector::load(
            self.context,
            vector_type,
            value,
            &[self.zero_index()],
            self.location,
        )
        .into();
        self.block()
            .append_operation(operation)
            .result(0)
            .unwrap()
            .into()
    }
}

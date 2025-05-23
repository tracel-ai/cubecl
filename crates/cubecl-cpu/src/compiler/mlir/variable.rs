use cubecl_core::ir::{Variable, VariableKind};
use melior::{
    dialect::{arith, memref},
    ir::{BlockLike, Type, Value, attribute::IntegerAttribute},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_variable(&mut self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            // VariableKind::GlobalInputArray(id) => {}
            // VariableKind::GlobalOutputArray(id) => {}
            VariableKind::LocalConst { id } => {
                let ptr_type = self.item_to_memref_type(variable.item);
                let value = self
                    .block
                    .append_operation(memref::alloca(
                        self.context,
                        ptr_type,
                        &[],
                        &[],
                        None,
                        self.location,
                    ))
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
    pub fn get_variable(&mut self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.global_buffers[id as usize]
            }
            VariableKind::LocalConst { id } => self
                .current_local_variables
                .get(&id)
                .expect("Variable should have been declared before")
                .clone(),
            VariableKind::Builtin(_builtin) => {
                let attribute = IntegerAttribute::new(Type::index(self.context), 0).into();
                self.block
                    .append_operation(arith::constant(self.context, attribute, self.location))
                    .result(0)
                    .unwrap()
                    .into()
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }
}

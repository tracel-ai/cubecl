use cubecl_core::ir::Operator;
use melior::{
    dialect::ods::vector,
    ir::{BlockLike, Value},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Value<'_, '_>) {
        match operator {
            Operator::Index(index_operator) => {
                let memref = self.get_variable(index_operator.list);
                let vector_type = self.item_to_type(index_operator.list.item);
                let index = self.get_variable(index_operator.index);
                let operation =
                    vector::load(self.context, vector_type, memref, &[index], self.location).into();
                let load_ssa = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                let index = self.zero_index();
                let operation =
                    vector::store(self.context, load_ssa, out, &[index], self.location).into();
                self.block().append_operation(operation);
            }
            Operator::IndexAssign(index_assign) => {
                let memref = self.get_variable_vector(index_assign.value);
                let index = self.get_variable(index_assign.index);
                let operation =
                    vector::store(self.context, memref, out, &[index], self.location).into();
                self.block().append_operation(operation);
            }
            _ => todo!("{} is not yet implemented", operator),
        }
    }
}

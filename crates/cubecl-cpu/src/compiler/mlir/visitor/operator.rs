use cubecl_core::ir::{Operator, Variable};
use melior::{dialect::ods::vector, ir::BlockLike};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::Index(index_operator) => {
                let memref = self.get_variable(index_operator.list);
                let vector_type = self.item_to_type(index_operator.list.item);
                let index = self.get_index(index_operator.index);
                let operation =
                    vector::load(self.context, vector_type, memref, &[index], self.location).into();
                let load_ssa = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                self.insert_variable(out, load_ssa);
            }
            Operator::IndexAssign(index_assign) => {
                let memref = self.get_variable(index_assign.value);
                let index = self.get_index(index_assign.index);
                let out = self.get_variable(out);
                let operation =
                    vector::store(self.context, memref, out, &[index], self.location).into();
                self.block().append_operation(operation);
            }
            _ => todo!("{} is not yet implemented", operator),
        }
    }
}

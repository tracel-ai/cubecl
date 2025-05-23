use cubecl_core::ir::Operation;
use melior::ir::Value;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Value<'_, '_>) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            _ => todo!("{} is not implemented yet.", operation),
        }
    }
}

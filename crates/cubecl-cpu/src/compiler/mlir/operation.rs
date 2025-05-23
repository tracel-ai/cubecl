use cubecl_core::ir::Operation;
use melior::ir::Value;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation_with_out(&self, operation: &'a Operation, out: Value<'a, 'a>) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            _ => todo!("Operation are not all implemented yet."),
        }
    }
}

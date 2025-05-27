use cubecl_core::ir::{Arithmetic, Operation, Variable};
use melior::{dialect::arith, ir::BlockLike};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Variable) {
        match operation {
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            Operation::Arithmetic(Arithmetic::Add(add)) => {
                let result = self
                    .block()
                    .append_operation(arith::addf(
                        self.get_variable(add.lhs),
                        self.get_variable(add.rhs),
                        self.location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();
                self.insert_variable(out, result);
            }
            _ => todo!("{} is not implemented yet.", operation),
        }
    }
}
